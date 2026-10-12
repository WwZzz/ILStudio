import vlastudio as vla


def main():
    policy = vla.load_policy("act", checkpoint="checkpoints/act_aloha")
    vla.serve(policy, address="0.0.0.0:5000")


if __name__ == "__main__":
    main()
