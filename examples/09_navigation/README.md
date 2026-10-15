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

## Learned VLN models

The completed full-split results, selected trained checkpoints and acceptance
audit are in the [navigation acceptance report](../../docs/evidence/navigation/paper/README.md).

[PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) fixes the full-split protocol and
paper references. CMA uses RGB, normalized depth, and original instruction tokens;
Recurrent VLN-BERT uses instruction text, current-viewpoint Places365 panorama
features, and simulator candidate geometry. Neither policy receives a goal,
reference path, expert action, or success metric.

Use `habitat.habitat_vlnce_cma` and `matterport3d.matterport_r2r_vlnbert` as the
environment templates, resolving dataset paths before running. The latter declares
the paper's 36-view candidate alignment and disables native image rendering because
the policy reads the authors' cached visual features. The simulator still performs
every turn and move on the real MP3D graph.

Convert published checkpoints with `convert_cma_checkpoint.py` or
`convert_vlnbert_checkpoint.py`, providing verified SHA256 hashes. Both use
restricted tensor loading. `prepare_r2r_features.py` verifies the author archive
before producing an indexed panorama array. Full evaluation is resumable:

```bash
python examples/09_navigation/evaluate_cma.py --config /path/cma-env.yaml \
  --model /path/cma-checkpoint --episodes 1839 --batch-size 8 \
  --output results/cma-val-unseen
python examples/09_navigation/evaluate_vlnbert.py --config /path/r2r-env.yaml \
  --model /path/vlnbert-checkpoint --batch-size 8 \
  --output results/vlnbert-val-unseen
```

`finetune_cma.py` and `finetune_vlnbert.py` perform small, train-only continuation
experiments and save separate checkpoints, loss histories and parameter changes.
They do not reproduce the papers' complete original training schedules. CMA checks
official training trajectory replay before learning, and maps episode IDs to the
native dataset order rather than assuming file order is preserved.

`collect_cma_dagger.py` additionally collects labels at learner-visited training
states, mixing the policy and a privileged 0.5 m shortest-path follower. It stores
the actions actually executed separately from expert labels. Reuse these samples
with `finetune_cma.py --precollected /path/collection/features`, keeping the same
model, train config, trajectory count and seed. The loader verifies collection
provenance and uses actual previous actions as recurrent inputs. The small native
DAgger gate validates alignment and backpropagation; it is not a benchmark result.

R2R's published metric scores the final viewpoint, including at the 15-decision
budget. Its config declares `success_requires_stop: false`, and also records
`explicit_stop_success` separately. The standalone paper evaluator uses the same
final-viewpoint rule, including for earlier configs without this field. VLN-CE
success requires an explicit STOP within 3 m. Do not compare the generic
environment's strict STOP metric to the paper's R2R metric without this distinction.
