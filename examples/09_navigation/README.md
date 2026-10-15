# Train a navigation policy

See [navigation setup and protocols](../../docs/navigation.md) for the Habitat-Sim
native build and the optional continuous R2R / Matterport3D adapters.

```bash
python examples/09_navigation/train_and_evaluate.py \
  --scene data/scene_datasets/habitat-test-scenes/skokloster-castle.glb \
  --output results/navigation
```

This trains a small GPS/compass PointNav policy in real Habitat simulation and
checks success rate >=30% and above a random-action baseline on 100 held-out
episodes. It saves a checkpoint, all episode definitions, per-episode actions,
metrics, training history and a rollout video. It does not claim R2R language
following or unseen-scene generalization.
