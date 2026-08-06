from pathlib import Path
from types import SimpleNamespace

import pytest

from configs.training.loader import TrainingConfig
from policy.trainer import BaseTrainer


def _training_args(tmp_path: Path, **config):
    values = {
        "report_to": [],
        "save_total_limit": 50,
        **config,
    }
    return TrainingConfig(values).to_training_arguments(
        SimpleNamespace(output_dir=str(tmp_path))
    )


def _bare_trainer(args):
    trainer = object.__new__(BaseTrainer)
    trainer.args = args
    trainer.state = SimpleNamespace(best_model_checkpoint=None)
    return trainer


def test_latest_only_maps_to_one_checkpoint(tmp_path):
    args = _training_args(tmp_path, save_latest_checkpoint_only=True)

    assert args.save_latest_checkpoint_only is True
    assert args.save_total_limit == 1


def test_latest_only_is_disabled_by_default(tmp_path):
    args = _training_args(tmp_path)

    assert args.save_latest_checkpoint_only is False
    assert args.save_total_limit == 50


def test_latest_only_rejects_best_model_tracking(tmp_path):
    with pytest.raises(ValueError, match="load_best_model_at_end"):
        _training_args(
            tmp_path,
            save_latest_checkpoint_only=True,
            load_best_model_at_end=True,
        )


def test_latest_only_deletes_older_checkpoints(tmp_path):
    args = _training_args(tmp_path, save_latest_checkpoint_only=True)
    for step in (10, 20, 30):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "complete").write_text("ok", encoding="utf-8")

    _bare_trainer(args)._rotate_checkpoints(output_dir=str(tmp_path))

    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-30"
    ]


def test_default_rotation_behavior_is_unchanged(tmp_path):
    args = _training_args(tmp_path)
    for step in (10, 20, 30):
        (tmp_path / f"checkpoint-{step}").mkdir()

    _bare_trainer(args)._rotate_checkpoints(output_dir=str(tmp_path))

    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-10",
        "checkpoint-20",
        "checkpoint-30",
    ]
