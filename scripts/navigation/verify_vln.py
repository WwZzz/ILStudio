"""Exercise real VLN assets and save expert replay plans for framework checks.

This tests simulator integration, not learned language following. Use the same
config and plans with eval_sim to check inference worker / SHM / action manager.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Fully resolved environment YAML')
    parser.add_argument('--episodes', type=int, default=6)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = SimpleNamespace(**yaml.safe_load(Path(args.config).read_text()))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    is_lab = getattr(config, 'backend', '') == 'lab'
    if is_lab:
        from vlastudio.benchmark.habitat import HabitatLabEnv
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
        env = HabitatLabEnv(config)
    else:
        from vlastudio.benchmark.matterport3d import MatterportNavigationEnv
        env = MatterportNavigationEnv(config)
    plans, records = {}, []
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            assert obs.raw_lang and obs.image.dtype == np.uint8
            assert obs.image.ndim == 4 and obs.image.shape[1] == 3 and np.ptp(obs.image) > 0
            if obs.depth is not None:
                assert np.isfinite(obs.depth).all() and obs.depth.ndim == 3
            if episode == 0:
                imageio.imwrite(output / 'rgb.png', obs.image[0].transpose(1, 2, 0))
                (output / 'observation_contract.json').write_text(json.dumps({
                    'image_shape': list(obs.image.shape),
                    'depth_shape': list(obs.depth.shape) if obs.depth is not None else None,
                    'action_mode': getattr(config, 'action_mode', 'primitive')}))
            actions = []
            def step(action):
                actions.append(int(action))
                return env.step(int(action))
            if is_lab:
                np.testing.assert_array_equal(obs.state, [0., 0.])
                follower = ShortestPathFollower(env.env.sim, .2, return_one_hot=False)
                while not env.done:
                    action = follower.get_next_action(env.env.current_episode.goals[0].position)
                    if action is None:
                        raise RuntimeError('No expert route found')
                    step(action)
            else:
                for target in env.episode['path'][1:]:
                    if env.navigation_options['action_mode'] == 'candidate_viewpoint':
                        step(env.viewpoint_actions[target])
                    else:
                        # Search discrete viewing angles until FORWARD selects
                        # the desired route edge. Never bypass the adapter.
                        moved = False
                        for level in range(3):
                            for turn in range(12):
                                locations = env.env.getState()[0].navigableLocations
                                if len(locations) > 1 and locations[1].viewpointId == target:
                                    step(1)
                                    moved = True
                                    break
                                step(3)
                            if moved:
                                break
                            step(5 if level == 0 else 4)
                            if level == 1:
                                step(4)
                        if not moved:
                            raise RuntimeError('Route edge cannot be reached with primitive FORWARD')
                    assert env.env.getState()[0].location.viewpointId == target
                step(0)
                assert env.travelled > 0, 'Path length must survive native in-place state changes'
            assert env.last_info['success'], env.last_info
            if obs.raw_lang in plans and plans[obs.raw_lang] != actions:
                raise RuntimeError('Ambiguous instruction cannot be used for diagnostic replay')
            plans[obs.raw_lang] = actions
            records.append({'instruction': obs.raw_lang, 'actions': actions, 'metrics': env.last_info})
            print('verified', episode, env.last_info, flush=True)
        env.reset()
        _, _, done, info = env.step(0)
        assert done and not info['success'], 'Early STOP must fail on these nontrivial R2R routes'
        _, _, repeated_done, repeated_info = env.step(0)
        assert repeated_done and repeated_info == info
    finally:
        env.close()
    report = {'purpose': 'Real-asset integration test with privileged expert; not learned VLN performance',
              'config': vars(config), 'episodes': records, 'early_stop': info}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'plans.json').write_text(json.dumps(plans, indent=2) + '\n')
    ctrl_type = getattr(config, 'ctrl_type', 'discrete')
    (output / 'config.json').write_text(json.dumps({'ctrl_space': 'navigation', 'ctrl_type': ctrl_type,
        'chunk_size': 1, 'state_dim': 2 if is_lab else 3, 'action_dim': 1}))
    (output / 'policy_metadata.json').write_text(json.dumps({
        'policy_module': str(Path(__file__).with_name('replay_policy.py').resolve()) + ':',
        'policy_name': 'diagnostic_expert_replay'}))
    (output / 'normalize.json').write_text(json.dumps({'datasets': [{
        'dataset_id': 'navigation_diagnostic', 'ctrl_space': 'navigation', 'ctrl_type': ctrl_type}],
        'state': {'navigation_diagnostic': 'identity'}, 'action': {'navigation_diagnostic': 'identity'}}))


if __name__ == '__main__':
    main()
