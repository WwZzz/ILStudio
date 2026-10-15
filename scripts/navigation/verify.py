"""Verify native PointNav and optional Habitat-Lab adapter on a real test scene.

The one-episode R2R-schema fixture verifies API wiring only, not VLN performance.
"""
import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vlastudio.benchmark.habitat import HabitatNavigationEnv, HabitatLabEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', default='results/navigation-verification')
    parser.add_argument('--lab', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(scene=args.scene, render=True, image_size=64, seed=51,
                             max_timesteps=80, success_distance=.2, min_distance=1., max_distance=5.)
    env = HabitatNavigationEnv(config)
    try:
        obs = env.reset()
        episode = dict(env.episode)
        assert obs.image.shape == (1, 3, 64, 64) and obs.image.dtype == np.uint8
        assert obs.depth.shape == (1, 64, 64) and np.isfinite(obs.depth).all()
        assert np.ptp(obs.image) > 0
        import imageio.v2 as imageio
        imageio.imwrite(output / 'habitat-rgb.png', obs.image[0].transpose(1, 2, 0))
        expert_actions = []
        while not env.done:
            action = env.expert_action()
            expert_actions.append(action)
            env.step(action)
        assert env.last_info['success'], env.last_info
        report = {'pointnav_expert': env.last_info, 'expert_actions': expert_actions,
                  'rgb_shape': list(obs.image.shape), 'depth_shape': list(obs.depth.shape)}
    finally:
        env.close()
    if args.lab:
        # Public test-scene fixture, with a real native renderer and geodesic scorer.
        # No Matterport data or official R2R benchmark episode is used here.
        fixture = output / 'r2r-api-fixture.json.gz'
        yaw = episode['yaw']
        record = {'episode_id': 'api-fixture-0', 'scene_id': str(Path(args.scene).resolve()),
                  'start_position': episode['start'],
                  'start_rotation': [0., float(np.sin(yaw / 2)), 0., float(np.cos(yaw / 2))],
                  'goals': [{'position': episode['goal']}],
                  'reference_path': [episode['start'], episode['goal']], 'trajectory_id': 0,
                  'instruction': {'instruction_text': 'Navigate to the goal.', 'instruction_tokens': [1]}}
        with gzip.open(fixture, 'wt') as stream:
            json.dump({'instruction_vocab': {'word_list': ['<pad>', 'goal']}, 'episodes': [record]}, stream)
        lab_config = SimpleNamespace(habitat_config='benchmark/nav/vln_r2r.yaml', habitat_overrides=[
            f'habitat.dataset.data_path={fixture}', 'habitat.dataset.scenes_dir=/',
            'habitat.simulator.turn_angle=30'])
        lab = HabitatLabEnv(lab_config)
        try:
            obs = lab.reset()
            assert obs.raw_lang == record['instruction']['instruction_text']
            np.testing.assert_array_equal(obs.state, [0., 0.])
            _, _, done, info = lab.step(0)
            assert done
            report['habitat_lab_api_fixture'] = {**info, 'instruction': obs.raw_lang,
                                                 'benchmark_performance': False}
        finally:
            lab.close()
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
