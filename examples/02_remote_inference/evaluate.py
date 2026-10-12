import vlastudio as vla


def main():
    policy = vla.connect_policy("127.0.0.1:5000")
    vla.load_env("aloha_transfer").evaluate(policy, output_dir="results/act_aloha_remote")


if __name__ == "__main__":
    main()
