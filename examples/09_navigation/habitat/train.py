"""Train navigation with VLAStudio's standard dataset and trainer interfaces."""
import argparse
from pathlib import Path
import vlastudio as vla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='checkpoints/example09/cma')
    parser.add_argument('--steps', type=int, help='Optional short training run')
    args = parser.parse_args()
    config = Path(__file__).resolve().parent/'config'
    dataset = vla.load_dataset(config/'task.yaml')
    policy = vla.load_policy('vlnce_cma')
    overrides = {'training.max_steps': args.steps} if args.steps else None
    vla.train(policy, dataset, config/'training.yaml', output_dir=args.output, overrides=overrides)


if __name__ == '__main__':
    main()
