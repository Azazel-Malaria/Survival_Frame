"""A prototype must never be consumed under another fold or protocol."""
import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils.experiment_config import file_sha256, prototype_metadata_path, validate_prototype


class PrototypeProvenanceTests(unittest.TestCase):
    def test_provenance_rejects_wrong_training_fold_protocol_and_modified_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.csv"
            train.write_text("case_id,slide_id\ncase1,slide1\n")
            proto = root / "prototypes.pkl"
            proto.write_bytes(pickle.dumps({"prototypes": [[0.0] * 768 for _ in range(16)]}))
            metadata = {"schema_version": 1, "train_sha256": file_sha256(train),
                        "feature_dir": str(root.resolve()), "n_proto": 16, "in_dim": 768,
                        "prototype_sha256": file_sha256(proto), "endpoint": "dss",
                        "split_mode": "train_test"}
            prototype_metadata_path(proto).write_text(json.dumps(metadata))
            validate_prototype(proto, train, root, endpoint="dss", split_mode="train_test")
            for kwargs in ({"endpoint": "os"}, {"split_mode": "train_val_test"}, {"in_dim": 1024}):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    validate_prototype(proto, train, root, **kwargs)
            original = proto.read_bytes()
            proto.write_bytes(pickle.dumps({"prototypes": [[0.0] * 768]}))
            metadata["prototype_sha256"] = file_sha256(proto)
            prototype_metadata_path(proto).write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                validate_prototype(proto, train, root)
            proto.write_bytes(original)
            metadata["prototype_sha256"] = file_sha256(proto)
            prototype_metadata_path(proto).write_text(json.dumps(metadata))
            train.write_text("case_id,slide_id\ncase2,slide2\n")
            with self.assertRaisesRegex(ValueError, "train_sha256"):
                validate_prototype(proto, train, root)
            train.write_text("case_id,slide_id\ncase1,slide1\n")
            proto.write_bytes(b"modified fixture")
            with self.assertRaisesRegex(ValueError, "prototype_sha256"):
                validate_prototype(proto, train, root)

    def test_missing_metadata_has_no_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            proto = Path(temporary) / "prototypes.pkl"
            proto.write_bytes(b"test fixture")
            with self.assertRaisesRegex(FileNotFoundError, "provenance"):
                validate_prototype(proto, Path(temporary) / "train.csv", temporary)


if __name__ == "__main__":
    unittest.main()
