# RL policy adapter architecture

The public policy boundary has three parts:

```text
MetaPolicy
    | existing normalization, meta2obs, act2meta
    v
BasePolicyAdapter
    | lifecycle and stable algorithm-facing methods
    v
ActionAdapter
    | action sampling, likelihoods, traces and replay conversion
    v
RL algorithm
```

`BasePolicyAdapter` is the default implementation of the stable
`MetaPolicyAdapter` contract. It contains no policy-name or algorithm-name
branches. Every loaded policy must arrive through ILStudio's existing
`MetaPolicy`; there is no separate direct/meta policy binding.

## ActionAdapter

An `ActionAdapter` owns only action semantics. Built-in implementations are:

- `GaussianActionAdapter` for one-step continuous policies;
- `CategoricalActionAdapter` for discrete policies;
- `GaussianChunkActionAdapter` for an outer Gaussian over action chunks;
- `NativeActionAdapter` for existing imitation-learning inference and loss.

Policy-native likelihoods live beside their policy. OpenVLA provides
`policy.openvla.rl_adapter.OpenVLAActionAdapter`, which samples action tokens
and recomputes their likelihoods with teacher forcing. ACT similarly provides
`policy.act.rl_adapter.ACTActionAdapter` for chunk actions and features.

Configuration selects the component by import path:

```yaml
policy_adapter:
  fallback_adapter: base
  required_capabilities: [action, evaluate_actions]
  args:
    action_adapter:
      type: rl.policy_adapter.action.GaussianActionAdapter
      args:
        initial_std: 0.02
        learn_std: true
```

OpenVLA uses the same base adapter:

```yaml
policy_adapter:
  required_capabilities: [action, recompute_traces]
  args:
    action_adapter:
      type: policy.openvla.rl_adapter.OpenVLAActionAdapter
      args:
        temperature: 1.6
```

Algorithms depend on capabilities such as `evaluate_actions`,
`sample_actions`, `action_scores`, or `recompute_traces`; they never inspect
the concrete policy or action-adapter type.

## Gradient boundary

Collection runs without a persistent computation graph and converts only the
sampled action to NumPy for the environment. The rollout stores actions and
likelihood traces. Updates preprocess the stored observations through the same
`MetaPolicy` chain and run the policy again with gradients enabled.

## Critics and algorithms

Critics remain independent configured modules. Algorithms own losses,
advantages, target networks, temperature parameters, exploration schedules
and update order. Adding an algorithm must not add a branch to either
`BasePolicyAdapter` or an `ActionAdapter`.
