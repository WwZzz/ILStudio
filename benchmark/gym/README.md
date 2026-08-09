# Gym benchmark

`benchmark.gym.GymEnv` adapts small Gymnasium environments for fast RL
validation. It is deliberately state-only: observations are flattened into
`MetaObs.state` and `MetaObs.image` is `None`.

The initial task pair covers both action families used by ILStudio algorithms:

- `gym.cartpole`: discrete actions for DQN, SARSA and discrete policy-gradient methods.
- `gym.pendulum`: continuous actions for DDPG, SAC and continuous policy-gradient methods.

Both tasks keep Gymnasium's original reward and five-value step API.
