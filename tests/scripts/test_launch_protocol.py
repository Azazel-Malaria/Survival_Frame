"""Protocol isolation and run aggregation tests without model/GPU dependencies."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("run_survival", REPO / "tools/run_survival.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from utils.experiment_config import CANCERS, MODELS, default_batch_size, load_data_paths, resolve_data_recipe, resolve_protocol


class ProtocolTests(unittest.TestCase):
    def test_default_and_validation_rules(self):
        plain = resolve_protocol()
        self.assertEqual(plain["result_group"], "DSS_standard_test")
        self.assertEqual(plain["split_names"], "train,test")
        for endpoint in ("dss", "os"):
            best = resolve_protocol(endpoint, checkpoint="best")
            early = resolve_protocol(endpoint, early_stopping=True)
            self.assertIn("val_test", best["split_root"])
            self.assertEqual(best["split_root"], early["split_root"])
            self.assertIn("earlystop", early["result_group"])
            self.assertNotEqual(best["result_group"], early["result_group"])

    def test_model_sources_and_batch_defaults(self):
        for model in MODELS:
            self.assertEqual(default_batch_size(model, "cox"), 64)
            self.assertEqual(default_batch_size(model, "nll"), 64 if model == "dimaf" else 1)
        for cancer in CANCERS:
            for model in MODELS:
                recipe = resolve_data_recipe(model, cancer)
                self.assertNotEqual(recipe["starpath_titan_model_path"],
                                    str(Path(recipe["titan_embeddings_path"]).parent))
                if model == "mcat":
                    self.assertIn("surv_set/raw_rna_data/combine", recipe["omics_path"])
                if model == "survpath":
                    self.assertIn("mmp_set/hallmarks", recipe["omics_path"])
        self.assertTrue(resolve_data_recipe("starpath", "CRC")["data_source"].endswith("/CRC"))
        self.assertTrue(resolve_data_recipe("starpath", "LUSC")["data_source"].endswith("/NSCLC"))
        self.assertTrue(resolve_data_recipe("starpath", "KIRC")["data_source"].endswith("/RCC"))
        with self.assertRaises(ValueError):
            resolve_data_recipe("dimaf", "BRCA", "surv_set")

    def test_configuration_changes_are_isolated(self):
        base = ["--cancer", "BRCA", "--model", "starpath", "--folds", "0", "--dry-run"]
        ids = set()
        for extra in ([], ["--loss", "cox"], ["--rna-set", "surv_set"],
                      ["--batch-size", "2"], ["--checkpoint", "best"],
                      ["--inject-layers", "3,4"], ["--trainable-layers", "4,5"],
                      ["--", "--starpath_num_regions", "16"]):
            args = runner.parser().parse_args(base + extra)
            _, _, run = runner.resolve_experiment(args)
            ids.add(run.parent.name)
        self.assertEqual(len(ids), 8)

    def test_managed_arguments_cannot_override_provenance(self):
        args = runner.parser().parse_args(["--cancer", "BRCA", "--model", "titan",
                                           "--", "--split_dir", "/wrong/fold"])
        with self.assertRaisesRegex(ValueError, "managed"):
            runner.resolve_experiment(args)

    def test_summary_is_current_run_only_and_sample_std(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for fold, value in [(0, .5), (1, .7), (4, .99)]:
                dest = run / f"fold_{fold}"
                dest.mkdir()
                (dest / "summary.csv").write_text(f"fold,c_index_test\n{fold},{value}\n")
            info = runner.aggregate_run(run, [0, 1])
            self.assertFalse(info["complete_five_fold_cv"])
            with (run / "cv_summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[-2]["c_index_test"]), .6)
            self.assertAlmostEqual(float(rows[-1]["c_index_test"]), 2 ** .5 * .1)
            self.assertEqual(info["std_ddof"], 1)


if __name__ == "__main__":
    unittest.main()
