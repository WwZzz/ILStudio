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
Policy -> MetaPolicy -> BasePolicyAdapter -> RLPolicyExecutor   RLRunner
                         |
                    ActionAdapter
   ^                                                   |
   +----- TrainerAdapter <- RLAlgorithm <--------+
```

- `benchmark/env_runner`: owns environment lifecycle. The initial collector is
  deliberately single-environment and synchronous. Environments that are not
  multiprocessing-safe therefore work without a hidden subprocess. A future
  vector/async runner is a separate component, not a flag inside benchmark
  adapters.
- `rl/policy_adapter`: composes the existing `MetaPolicy` with one configured
  `ActionAdapter`. Policy-local modules provide only action semantics such as
  OpenVLA token likelihoods, not parallel RL pipelines.
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
- `rl/policy_adapter/training/optimizer.py`: owns backward/optimizer mechanics or delegates them to a
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

### Online, offline, and hybrid data modes

The outer runner selects where replay data comes from; algorithms and policy
adapters never branch on the mode:

```text
online:  environment collection -> replay/rollout -> update
offline: task dataset -> lazy offline replay -> update
hybrid:  task dataset + online replay -> ratio-controlled mixed update
```

Pure offline training requires a replay-based algorithm and does not construct
an environment, executor, reward composer, or collector:

```bash
.venv/bin/python train_rl.py \
  -m ckpt/my_policy \
  --mode offline \
  -t configs/task/d4rl/door_expert.yaml \
  -a td3_bc \
  --runner default \
  -c rl \
  -o ckpt/my_policy_offline
```

Hybrid mode keeps the task dataset read-only, writes new experience to a
separate online replay buffer, and samples both sources without allowing
collection rejection or buffer clearing to erase demonstrations:

```bash
.venv/bin/python train_rl.py \
  -m ckpt/my_policy \
  --mode hybrid \
  -t configs/task/my_demonstrations.yaml \
  -e metaworld.easy00 \
  -a td3 \
  --offline-ratio 0.75 \
  --offline-pretrain-iterations 1000 \
  -o ckpt/my_policy_hybrid
```

Task samples keep the existing supervised fields. RL additionally recognizes
optional `success`, `reward`, `terminated`, and `truncated` fields. If an
episode has no success label, it is treated as a successful demonstration by
default: only its terminal transition gets `success=true` and a default sparse
`train/total=1` reward. Use `--offline-missing-success failure` to change that
policy. These defaults are applied by `rl.offline.OfflineReplayDataset`; the
samples returned to `train.py` are never modified.

Offline replay is lazy: images and transitions are converted only for sampled
indices rather than copied into RAM. Raw environment-space state and action are
stored in the RL contract, while the checkpoint's saved `MetaPolicy`
normalizers are still applied in policy and action-adapter forwards.

The continuous-action offline presets cover three different distribution-shift
strategies:

| Config | Main idea | Compatible policy |
| --- | --- | --- |
| `iql` | expectile V plus advantage-weighted log likelihood | stochastic one-step policy |
| `iql_act` | the same IQL targets with fixed-variance-equivalent weighted chunk MSE | ACT action chunks |
| `cql` | SAC plus conservative log-sum-exp Q regularization | stochastic bounded one-step policy |
| `td3_bc` | TD3 with batch-normalized Q maximization and behavior cloning | deterministic ACT action chunks |

For example, ACT demonstrations use:

```bash
.venv/bin/python train_rl.py \
  -m ckpt/my_act_policy \
  --mode offline \
  -t configs/task/my_demonstrations.yaml \
  -a iql_act \
  --runner default \
  -c rl \
  -o ckpt/my_act_iql
```

`cql` intentionally requires an action adapter that can both sample policy
actions with log probabilities and sample uniformly from its bounded action
space. It therefore fails capability validation instead of silently applying a
different loss to deterministic ACT outputs.

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

The policy output is the mean of a Gaussian in the checkpoint's normalized
action space. By default, `policy_std` initializes one state-independent
standard deviation per action dimension; their logarithms are optimized
together with the policy;
`learn_fixed_std: false` keeps them frozen for controlled debugging. The adapter
records the sampled action
and per-step log probability under `PolicyOutput.policy_info` before the shared
action-chunk manager dispatches it. During an update it reconstructs the chunk
from the original `MetaObs`; stochastic base policies are replayed with their
stored per-decision torch seed, so gradients reach the original policy.

`policy_std` is the initial standard deviation in normalized action space and
must be tuned for the checkpoint and controller. In particular, absolute end-effector control can
require a much smaller value than delta control. The executed action still goes
through the checkpoint's normalizer, optional post-processing, and benchmark
`MetaEnv.meta2act` constraints.

This reusable adapter defines an outer Gaussian over the final diffusion-policy
chunk; it is not DPPO's per-denoising-step likelihood. PPO obtains values from
its separately configured critic; the action adapter never owns a value head.
A trace adapter is still required for token likelihood or denoising-step
objectives. Sparse environments also need a
successful rollout or an additional reward module: an all-zero rollout
correctly produces no policy update.

## Built-in algorithms

Built-ins are selected by files under `configs/rl/algorithm`; each file names
the algorithm import path, buffer, adapter capabilities and update defaults.
There is no algorithm switch in `train_rl.py`.

| Algorithm | Component type | Buffer | Policy-adapter capabilities | Optimizer keys |
| --- | --- | --- | --- | --- |
| REINFORCE | `rl.algorithm.ReinforceAlgorithm` | rollout | `action`, `evaluate_actions` | one optimizer |
| Actor-Critic | `rl.algorithm.ActorCriticAlgorithm` | rollout | `action`, `evaluate_actions` | one optimizer |
| PPO | `rl.algorithm.PPOAlgorithm` | rollout | `evaluate_actions` or `recompute_traces` | one optimizer |
| DQN / Double-DQN | `rl.algorithm.DQNAlgorithm` | replay | `action`, `action_scores` | one optimizer |
| SARSA | `rl.algorithm.SARSAAlgorithm` | rollout | `action`, `action_scores` | one optimizer |
| DDPG | `rl.algorithm.DDPGAlgorithm` | replay | `action`, `sample_actions`, `batch_actions` | `critic`, `actor` |
| SAC | `rl.algorithm.SACAlgorithm` | replay | `action`, `sample_actions`, `batch_actions` | `critic1`, `critic2`, `actor`; optional `alpha` |
| IQL | `rl.algorithm.IQLAlgorithm` | replay | `batch_actions` plus `evaluate_actions` or `sample_actions` | `critic1`, `critic2`, `value`, `actor` |
| CQL | `rl.algorithm.CQLAlgorithm` | replay | `sample_actions`, `batch_actions`, `uniform_actions` | `critic1`, `critic2`, `actor`; optional `alpha` |
| TD3+BC | `rl.algorithm.TD3BCAlgorithm` | replay | `sample_actions`, `batch_actions` | `critic`, `actor` |

Algorithms call explicit distribution or tensor operations. They own returns,
TD/GAE targets, clipping, critics, target networks, temperature, exploration
schedules, and post-update maintenance. Policy adapters never branch on an
algorithm name.

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

The default `meta_policy` adapter advertises only `action`/`training_forward`, so
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
