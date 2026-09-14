"""Validation-based selection and stopping, with explicit checkpoint provenance."""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

import torch


class CheckpointManager:
    def __init__(self, directory, *, selection="last", metric="c_index",
                 early_stopping=False, es_metric="loss", patience=5,
                 min_epochs=3, config=None):
        if selection not in {"last", "best"}:
            raise ValueError("checkpoint selection must be last or best")
        if metric not in {"loss", "c_index"} or es_metric not in {"loss", "c_index"}:
            raise ValueError("checkpoint metrics must be loss or c_index")
        if patience <= 0 or min_epochs <= 0:
            raise ValueError("patience and min_epochs must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.selection, self.metric = selection, metric
        self.early_stopping, self.es_metric = bool(early_stopping), es_metric
        self.patience, self.min_epochs = int(patience), int(min_epochs)
        self.config = dict(config or {})
        self.run_id = uuid.uuid4().hex
        self.best_epoch = self.best_score = self.es_best_score = None
        self.stopped_epoch = None
        self.counter = 0

    @staticmethod
    def _score(metrics, name):
        if name not in metrics:
            raise ValueError(f"Validation did not produce {name}")
        score = float(metrics[name])
        if not math.isfinite(score):
            raise ValueError(f"Validation {name} must be finite, received {score}")
        return score

    @staticmethod
    def _improves(score, previous, metric, *, ties=False):
        if previous is None:
            return True
        if metric == "loss":
            return score <= previous if ties else score < previous
        return score >= previous if ties else score > previous

    def _save(self, name, model, epoch, score=None):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"model": model.state_dict(), "epoch": int(epoch),
                    "score": score, "run_id": self.run_id, "config": self.config}, temporary)
        temporary.replace(path)

    def step(self, epoch, model, val_metrics):
        """Track validation best independently of the early-stopping counter."""
        score = self._score(val_metrics, self.metric)
        # SlotSPE's released selection saves the later epoch on a C-index tie.
        if self._improves(score, self.best_score, self.metric, ties=True):
            self._save("best_checkpoint.pth", model, epoch, score)
            self.best_epoch, self.best_score = int(epoch), score
        if not self.early_stopping:
            return False
        stop_score = self._score(val_metrics, self.es_metric)
        if self._improves(stop_score, self.es_best_score, self.es_metric):
            self.es_best_score, self.counter = stop_score, 0
        else:
            self.counter += 1
        if epoch + 1 >= self.min_epochs and self.counter >= self.patience:
            self.stopped_epoch = int(epoch)
            return True
        return False

    def finish(self, model, last_epoch):
        """Save actual last epoch, then load the requested checkpoint for testing."""
        if last_epoch is None:
            raise ValueError("No epoch was trained")
        self._save("last_checkpoint.pth", model, last_epoch)
        selected_epoch = self.best_epoch if self.selection == "best" else int(last_epoch)
        if selected_epoch is None:
            raise RuntimeError("best checkpoint requires finite validation metrics from this run")
        path = self.directory / f"{self.selection}_checkpoint.pth"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("run_id") != self.run_id or state.get("epoch") != selected_epoch:
            raise RuntimeError("Checkpoint provenance or selected epoch does not match this run")
        if self.selection == "best" and state.get("score") != self.best_score:
            raise RuntimeError("Checkpoint score does not match the selected validation score")
        model.load_state_dict(state["model"], strict=True)
        record = {
            "checkpoint_selection": self.selection, "checkpoint_metric": self.metric,
            "checkpoint": path.name, "selected_epoch": selected_epoch,
            "best_epoch": self.best_epoch, "best_score": self.best_score,
            "last_epoch": int(last_epoch), "epoch_index_base": 0,
            "early_stopping": self.early_stopping, "es_metric": self.es_metric,
            "es_best_score": self.es_best_score, "stopped_epoch": self.stopped_epoch,
            "stop_reason": "patience" if self.stopped_epoch is not None else "max_epochs",
            "run_id": self.run_id,
        }
        (self.directory / "checkpoint_selection.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        return record
