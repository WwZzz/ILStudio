"""Continue TCP training on official Bench2Drive demonstrations."""
import argparse
from pathlib import Path
import vlastudio as vla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='checkpoints/example10/tcp')
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    config = Path(__file__).resolve().parent/'config'
    dataset = vla.load_dataset(config/'task.yaml')
    policy = vla.load_policy('tcp_b2d')
    overrides = {'training.max_steps': args.steps} if args.steps else None
    vla.train(policy, dataset, config/'training.yaml', output_dir=args.output, overrides=overrides)


if __name__ == '__main__':
    main()
