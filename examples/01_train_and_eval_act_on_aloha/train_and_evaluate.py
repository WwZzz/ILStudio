import vlastudio as vla
from pathlib import Path


def main():
    dataset = vla.load_dataset("sim_transfer_cube_scripted")
    policy = vla.load_policy("act")
    config = Path(__file__).with_name("training_config.yaml")
    vla.train(policy, dataset, config, output_dir="checkpoints/act_aloha_full")
    vla.load_env("aloha_transfer").evaluate(policy, output_dir="results/act_aloha_full", num_rollout=50)


if __name__ == "__main__":
    main()
