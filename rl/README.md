# ILStudio RL

The RL package is an additive training path built on ILStudio's existing
`MetaObs`, `MetaAction`, `MetaEnv`, policy loaders, and benchmark adapters. It
does not replace `eval_sim.py`, `eval_real.py`, `BasicActionManager`, or the SHM
inference process.

## Runtime flow

```text
benchmark MetaEnv -> EnvRunner -> Collector -> RewardComposer -> Buffer
                              ^                         |
                              |                         v
Policy -> PolicyAdapter -> RLPolicyExecutor       RLRunner
   ^                                                   |
   +----- TrainerAdapter <- RLAlgorithm <--------+
```

- `benchmark/env_runner`: owns environment lifecycle. The initial collector is
  deliberately single-environment and synchronous. Environments that are not
  multiprocessing-safe therefore work without a hidden subprocess. A future
  vector/async runner is a separate component, not a flag inside benchmark
  adapters.
- `rl/policy_adapter`: first looks for `policy/<name>/rl_adapter.py`; otherwise
  a reusable adapter from `rl/policy_adapter` can be selected explicitly.
- `rl/executor`: directly calls the current trainable policy and reuses the pure
  `deploy.action_manager.chunk.BasicActionChunkManager`. It remains independent
  of `BasicActionManager`, SHM inference, and the eval facade.
- `rl/collector`: creates `MetaTransition` objects and never hides terminated
  versus truncated episode boundaries.
- `rl/reward`: preserves raw environment reward as `env/raw`, combines
  namespaced reward modules, and writes the optimized sum as `train/total`.
- `rl/buffer`: shares storage and state-dict logic between rollout and replay
  buffers while retaining their different lifecycle rules.
- `rl/runner`: composes collection, sampling, algorithm updates, and callbacks.
  It owns lifecycle counters but never performs backward or optimizer steps.
- `rl/algorithm`: owns RL mathematics and declares required policy capabilities
  and buffer family.
- `rl/policy_adapter/trainer.py`: owns backward/optimizer mechanics or delegates them to a
  policy-specific update hook. `Trainer` therefore only names parameter-updating
  objects, consistent with existing `policy/*/trainer.py` classes.
  `build_trainer_adapter_from_components` prefers a policy-local RL adapter,
  then an explicit `Trainer.build_trainer_adapter` hook, then the generic optimizer
  adapter; it never starts a policy-specific full SFT `Trainer.train()` loop.
- `RLRunner.state_dict()` aggregates non-weight RL state for callbacks that use
  the existing ILStudio policy and project `ckpt/` workflow; RL does not own a
  second checkpoint storage manager.

## Unified entrypoint

`train_rl.py` is the canonical RL training entrypoint, aligned with `train.py`.
`-m/--model_name_or_path` always means a local ILStudio checkpoint whose model
parameters will continue training; remote inference servers and dummy policies
are rejected because they cannot expose local trainable parameters.

```bash
.venv/bin/python train_rl.py \
  -m ckpt/my_policy \
  -e metaworld.easy00 \
  -a ppo \
  -r raw \
  --env_runner sync \
  -c rl \
  -o ckpt/my_policy_ppo
```

Config choices are independent native ILStudio YAML categories:

```text
configs/rl/algorithm/ppo.yaml
configs/rl/algorithm/reinforce.yaml
configs/rl/reward/raw.yaml
configs/rl/env_runner/sync.yaml
configs/rl/runner/default.yaml
configs/training/rl.yaml
```

`train_rl.py` creates one `ConfigLoader`, just like `train.py`, and loads the
environment plus every selected fragment through it. The fragments are then
composed into the ordered runtime graph; the entrypoint contains no policy or
algorithm switch. Existing dotted overrides use the fragment category as their
root:

```bash
.venv/bin/python train_rl.py -m ckpt/my_policy -a ppo \
  --algorithm.args.clip_ratio 0.1 \
  --runner.args.iterations 3 \
  --training.learning_rate 0.000002 \
  --env.args.max_timesteps 128
```

Rewards compose by repeating `-r`; raw environment reward is always preserved
as `env/raw`, while each additional config contributes namespaced modules and
weights to one `RewardComposer`:

```bash
.venv/bin/python train_rl.py -m ckpt/my_policy \
  -r raw -r world_model
```

Each algorithm fragment declares its required policy capabilities and a generic
fallback adapter. Resolution is:

```text
policy/<checkpoint-policy-type>/rl_adapter.py
  -> algorithm-selected generic fallback (only when the local file is absent)
```

A policy-local adapter therefore wins automatically for Pi0, token policies,
or policy-specific RL variants. If it exists but is broken or lacks a required
capability, construction fails instead of silently changing the algorithm.
Token-level methods can place token log-probabilities, masks, and versions in
`PolicyOutput.policy_info`; collector and buffers preserve that metadata.

`configs/training/rl.yaml` uses the same native training loader and field names
as supervised training. RL currently maps only the fields with identical
semantics: `learning_rate`, `weight_decay`, Adam betas/epsilon, optimizer name,
and `seed`. SFT epochs, dataloaders, per-device batches, logging, scheduler and
checkpoint cadence are not reused; those belong to the RL runner and its
components.

### Continuous action-chunk policies

The generic `gaussian_chunk` fallback composes an existing local `MetaPolicy`
checkpoint, including ACT and diffusion policy, with REINFORCE or PPO when that
policy has no local RL adapter:

```bash
.venv/bin/python train_rl.py \
  -m ckpt/my_policy \
  -e metaworld.easy00 \
  -a reinforce \
  -r raw \
  -o ckpt/my_policy_reinforce
```

The policy output is the mean of a fixed-standard-deviation Gaussian in the
checkpoint's normalized action space. The adapter records the sampled action
and per-step log probability under `PolicyOutput.policy_info` before the shared
action-chunk manager dispatches it. During an update it reconstructs the chunk
from the original `MetaObs`; stochastic base policies are replayed with their
stored per-decision torch seed, so gradients reach the original policy.

`policy_std` is measured in normalized action space and must be tuned for the
checkpoint and controller. In particular, absolute end-effector control can
require a much smaller value than delta control. The executed action still goes
through the checkpoint's normalizer, optional post-processing, and benchmark
`MetaEnv.meta2act` constraints.

This reusable adapter defines an outer Gaussian over the final diffusion-policy
chunk; it is not DPPO's per-denoising-step likelihood. It supplies a zero value
baseline, so the provided PPO config deliberately sets `value_coef: 0.0`. A
policy-local adapter should replace it when a learned critic, token likelihood,
or denoising-step objective is required. Sparse environments also need a
successful rollout or an additional reward module: an all-zero rollout
correctly produces no policy update.

## Built-in algorithms

Built-ins are selected by files under `configs/rl/algorithm`; each file names
the algorithm import path, buffer, adapter capabilities and update defaults.
There is no algorithm switch in `train_rl.py`.

| Algorithm | Component type | Buffer | Policy-adapter capabilities | Optimizer keys |
| --- | --- | --- | --- | --- |
| REINFORCE | `rl.algorithm.ReinforceAlgorithm` | rollout | `action`, `reinforce` | one optimizer |
| Actor-Critic | `rl.algorithm.ActorCriticAlgorithm` | rollout | `action`, `actor_critic` | one optimizer |
| PPO | `rl.algorithm.PPOAlgorithm` | rollout | `action`, `ppo` | one optimizer |
| DQN / Double-DQN | `rl.algorithm.DQNAlgorithm` | replay | `action`, `dqn`, `target_update` | one optimizer |
| SARSA | `rl.algorithm.SARSAAlgorithm` | rollout | `action`, `sarsa` | one optimizer |
| DDPG | `rl.algorithm.DDPGAlgorithm` | replay | `action`, `ddpg`, `target_update` | `critic`, `actor` |
| SAC | `rl.algorithm.SACAlgorithm` | replay | `action`, `sac`, `target_update`; plus `temperature` for auto-alpha | `critic1`, `critic2`, `actor`; optional `alpha` |

`PolicyAdapter.algorithm_forward(operation, batch, context=...)` is the thin
boundary for model-specific tensorization and head selection. The algorithm
classes construct returns, TD/GAE targets, clipping, entropy terms and losses.
DDPG and SAC request a fresh actor forward after critic updates, then invoke
`algorithm_post_step` for target-network maintenance.

Multiple optimizers remain ordinary graph components and are injected into the
single trainer-adapter file as a mapping:

```yaml
trainer_adapter:
  type: rl.builders.build_trainer_adapter_from_components
  args:
    policy_components: {$ref: policy_components}
    optimizer:
      critic: {$ref: critic_optimizer}
      actor: {$ref: actor_optimizer}
```

The default `meta_policy` adapter advertises only `action`/`chunk_training`, so
incompatible algorithms fail during `RLRunner` construction. A policy becomes
algorithm-capable only when its local `policy/<name>/rl_adapter.py` or an
explicit reusable adapter such as `gaussian_chunk` supplies the corresponding
operation and required model outputs.

## Parallel environment policy

Do not infer subprocess safety from `num_envs`. Begin with `SyncEnvRunner` for a
new benchmark. Add a vector/async runner only after the benchmark has an
isolated reset/step/close stress test under its intended start method. LIBERO,
GUI simulators, and real robots may require main-process or serial execution.
The component graph permits a benchmark-specific runner without changing the
collector, policy, algorithm, or `train_rl.py` contracts.
