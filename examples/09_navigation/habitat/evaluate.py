"""Evaluate the navigation policy on the complete validation split."""
from pathlib import Path
from vlastudio.benchmark.habitat.evaluate_cma import main


if __name__ == '__main__':
    main(default_config=str(Path(__file__).resolve().parent/'config/val_unseen.yaml'))
