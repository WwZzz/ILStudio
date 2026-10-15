"""Shared navigation contract: discrete actions, explicit success and SPL."""
from dataclasses import asdict
import time

import numpy as np

from vlastudio.benchmark.base import MetaAction


def navigation_options(config, *, supports_candidates=False):
    """Declare control semantics separately from optional simulator affordances."""
    mode = getattr(config, 'action_mode', 'primitive')
    source = getattr(config, 'candidate_source', 'none')
    scope = getattr(config, 'candidate_scope', 'visible')
    if mode not in ('primitive', 'candidate_viewpoint'):
        raise ValueError(f'Unknown navigation action_mode: {mode}')
    if source not in ('none', 'simulator'):
        raise ValueError('candidate_source must be none or simulator; learned proposals belong to the policy')
    if scope not in ('visible', 'all_neighbors'):
        raise ValueError('candidate_scope must be visible or all_neighbors')
    if not supports_candidates and (mode != 'primitive' or source != 'none'):
        raise ValueError('This backend supports primitive actions and candidate_source=none only')
    if mode == 'candidate_viewpoint' and source != 'simulator':
        raise ValueError('candidate_viewpoint actions require candidate_source=simulator')
    if mode == 'primitive' and source == 'simulator' and scope != 'visible':
        raise ValueError('Primitive actions require visible candidates to preserve FORWARD semantics')
    ctrl_type = 'discrete' if mode == 'primitive' else 'candidate_viewpoint'
    if getattr(config, 'ctrl_type', ctrl_type) != ctrl_type or getattr(config, 'ctrl_space', 'navigation') != 'navigation':
        raise ValueError(f'{mode} requires ctrl_space=navigation, ctrl_type={ctrl_type}')
    return {'action_mode': mode, 'ctrl_type': ctrl_type, 'candidate_source': source,
            'candidate_scope': scope if source == 'simulator' else None, 'stop_action': 0}


def discrete_action(action, count, *, ctrl_type='discrete'):
    if isinstance(action, (MetaAction, dict)):
        if action['ctrl_space'] != 'navigation' or action['ctrl_type'] != ctrl_type:
            raise ValueError(f'Navigation requires ctrl_space=navigation, ctrl_type={ctrl_type}')
        action = action['action']
    value = np.asarray(action)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError('Expected one finite discrete navigation action')
    scalar = float(value.reshape(-1)[0])
    if scalar != int(scalar) or not 0 <= scalar < count:
        raise ValueError(f'Navigation action must be an integer in [0, {count})')
    return int(scalar)


def spl(success, shortest, travelled):
    if not success:
        return 0.0
    if shortest <= 1e-8:
        return float(travelled <= 1e-8)
    return float(shortest / max(shortest, travelled))


def evaluate(args, env, action_manager, inference_ctx=None, video_writer=None,
             save_example_dir=None):
    """VLAStudio evaluator; termination and timeout never imply success."""
    from vlastudio.benchmark.utils import organize_obs
    started = time.time()
    count = len(env)
    finished = np.zeros(count, dtype=bool)
    successes = np.zeros(count, dtype=bool)
    horizons = np.full(count, args.max_timesteps, dtype=int)
    metrics = [{} for _ in range(count)]
    try:
        obs = organize_obs(env.reset())
        for step in range(args.max_timesteps):
            if inference_ctx is not None:
                inference_ctx.update_obs(obs, step)
            actions = action_manager.select_action()
            raw_obs, _, done, info = env.step(actions)
            for i in range(count):
                if finished[i]:
                    continue
                metrics[i] = dict(info[i])
                successes[i] = bool(info[i].get('success', False))
                if done[i]:
                    finished[i] = True
                    horizons[i] = step + 1
            if video_writer is not None:
                for item in raw_obs:
                    images = item.image if hasattr(item, 'image') else item.get('image')
                    if images is not None:
                        video_writer.append_data(images[0].transpose(1, 2, 0))
            obs = organize_obs(raw_obs)
            if finished.all():
                break
    finally:
        if not getattr(args, 'reuse_env', False):
            env.close()
    return {
        'success': successes.tolist(), 'total_success': int(successes.sum()),
        'total': count, 'success_rate': float(successes.mean()),
        'spl': float(np.mean([m.get('spl', 0.0) for m in metrics])),
        'horizon': horizons.tolist(),
        'horizon_success': float(horizons[successes].mean()) if successes.any() else 0.0,
        'episodes': metrics, 'eval_time': time.time() - started,
    }


class NavigationEpisode:
    """Sticky terminal state makes finished vector slots safe to step again."""
    def _begin(self, shortest):
        self.steps = 0
        self.travelled = 0.0
        self.shortest = float(shortest)
        self.done = False
        self.last_info = {}

    def _finish_step(self, distance, stopped, collision=False):
        self.steps += 1
        self.done = bool(stopped or self.steps >= self.max_steps)
        success = bool(stopped and distance < self.success_distance)
        self.last_info = {
            'success': success, 'spl': spl(success, self.shortest, self.travelled),
            'distance_to_goal': float(distance), 'path_length': self.travelled,
            'shortest_path_length': self.shortest, 'steps': self.steps,
            'stopped': bool(stopped), 'timeout': bool(self.done and not stopped),
            'collision': bool(collision), 'episode_id': str(self.episode_id),
        }
        return asdict(self.prev_obs), float(success), self.done, dict(self.last_info)
