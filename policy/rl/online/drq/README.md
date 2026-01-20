# DrQ (Data-regularized Q)

DrQ is an image-based reinforcement learning algorithm that achieves state-of-the-art performance on DMC (DeepMind Control Suite) by using random shift augmentation for data regularization.

## Reference

Kostrikov et al., "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels" (2020)

- Paper: https://arxiv.org/abs/2004.13649
- Original Code: https://github.com/denisyarats/drq

## Key Features

1. **SAC-based**: Uses Soft Actor-Critic as the base algorithm
2. **Random Shift Augmentation**: Key to DrQ's performance - pads images and randomly crops
3. **Shared Encoder**: Actor and critic share convolutional weights
4. **Automatic Temperature Tuning**: SAC's entropy coefficient is learned

## Installation

DrQ requires the following additional dependencies:

```bash
# DMC environment wrapper
pip install dmc2gym

# For efficient augmentation (optional but recommended)
pip install kornia

# DeepMind Control Suite
pip install dm_control
```

## Usage

### Training

```bash
# Train DrQ on cartpole swingup
python train_rl.py -p drq -e dmc --env.task cartpole_swingup -o ckpt/drq_cartpole

# Train on cheetah run (harder task)
python train_rl.py -p drq -e dmc --env.task cheetah_run -o ckpt/drq_cheetah

# With custom hyperparameters
python train_rl.py -p drq -e dmc --env.task walker_walk \
    --training.lr 1e-4 \
    --training.batch_size 256 \
    --training.num_train_steps 500000
```

### Evaluation

```bash
# Evaluate trained model
python eval_sim.py -m ckpt/drq_cartpole -e dmc --env.task cartpole_swingup

# With video recording
python eval_sim.py -m ckpt/drq_cheetah -e dmc --env.task cheetah_run --save_video
```

## Configuration

### Policy Config (`configs/policy/drq.yaml`)

```yaml
type: policy.rl.online.drq

model_args:
  feature_dim: 50      # Encoder output dimension
  hidden_dim: 1024     # MLP hidden size
  hidden_depth: 2      # Number of hidden layers

training_args:
  discount: 0.99       # Gamma
  lr: 1e-3             # Learning rate
  batch_size: 512
  image_pad: 4         # Random shift padding
```

### Environment Config (`configs/env/dmc.yaml`)

```yaml
type: benchmark.dmc

task: cartpole_swingup
image_size: 84
action_repeat: 2
frame_stack: 3
```

## Available DMC Tasks

| Domain | Tasks |
|--------|-------|
| cartpole | balance, balance_sparse, swingup, swingup_sparse |
| cheetah | run |
| walker | stand, walk, run |
| hopper | stand, hop |
| finger | spin, turn_easy, turn_hard |
| ball_in_cup | catch |
| reacher | easy, hard |
| quadruped | walk, run |
| humanoid | stand, walk, run |
| pendulum | swingup |
| acrobot | swingup, swingup_sparse |

## Architecture

```
DrQAgent
├── Actor (policy network)
│   ├── Encoder (4 conv layers → feature_dim)
│   └── MLP (feature_dim → 2*action_dim for mu and log_std)
│
├── Critic (Q-networks)
│   ├── Encoder (shared conv weights with Actor)
│   ├── Q1 MLP (feature_dim + action_dim → 1)
│   └── Q2 MLP (feature_dim + action_dim → 1)
│
└── Critic Target (soft updated copy of Critic)
```

## Training Loop

1. **Seed Phase**: Random actions for `num_seed_steps` to fill buffer
2. **Training Phase**:
   - Sample action from policy
   - Execute in environment
   - Add transition to replay buffer
   - Sample batch with augmentation
   - Update critic (with augmented obs and next_obs)
   - Update actor (every `actor_update_frequency` steps)
   - Soft update target network

## Performance Tips

1. **Action Repeat**: Use `action_repeat=2` for most tasks (4 for finger tasks)
2. **Frame Stack**: 3 frames is standard
3. **Image Size**: 84x84 is standard; larger may help for complex tasks
4. **Batch Size**: 512 works well; can reduce to 256 for memory constraints
5. **Seed Steps**: 1000 is sufficient for most tasks





