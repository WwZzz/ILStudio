"""Train FastWAM using VLAStudio's standard HDF5 data and BaseTrainer."""
import argparse
import os
from pathlib import Path

import vlastudio as vla


def main():
    example = Path(__file__).resolve().parent
    cache = Path(os.environ.get("VLASTUDIO_CACHE", "~/.cache/vlastudio")).expanduser().resolve()
    os.environ["VLASTUDIO_CACHE"] = str(cache)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(cache / "models/fastwam"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="checkpoints/fastwam")
    parser.add_argument("--data", type=Path, default=cache / "data/fastwam/libero_hdf5")
    parser.add_argument("--steps", type=int, default=21700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    args = parser.parse_args()
    os.environ["VLASTUDIO_FASTWAM_DATA"] = str(args.data.resolve())
    policy = vla.load_policy(example / "config/train_policy.yaml")
    dataset = vla.load_dataset(example / "config/task.yaml")
    overrides = {
        "policy.args.dataset_stats": str(args.data.resolve() / "dataset_stats.json"),
        "training.max_steps": args.steps,
        "training.per_device_train_batch_size": args.batch_size,
        "training.gradient_accumulation_steps": args.accumulation,
        "training.deepspeed": str(example / "config/zero1.json"),
    }
    vla.train(policy, dataset, example / "config/training.yaml",
              output_dir=args.output_dir, overrides=overrides)


if __name__ == "__main__":
    main()
