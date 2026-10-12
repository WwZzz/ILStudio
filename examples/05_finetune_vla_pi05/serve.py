import vlastudio as vla


def main():
    policy = vla.load_policy(
        "examples/05_finetune_vla_pi05/pi05.yaml",
        checkpoint="checkpoints/pi05_tabletop",
    )
    vla.serve(policy, address="0.0.0.0:5000")


if __name__ == "__main__":
    main()
