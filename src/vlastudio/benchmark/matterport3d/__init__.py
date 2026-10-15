"""Classic R2R on the real MatterSim renderer and connectivity graphs."""
import gzip
import heapq
import json
from pathlib import Path

import numpy as np

from vlastudio.benchmark.base import MetaEnv, MetaObs
from vlastudio.benchmark.navigation import NavigationEpisode, discrete_action, evaluate


def create_env(config):
    return MatterportNavigationEnv(config)


def load_graph(path):
    nodes = json.loads(Path(path).read_text())
    positions = {n['image_id']: np.asarray(n['pose'])[ [3, 7, 11] ].astype(float)
                 for n in nodes if n['included']}
    graph = {key: {} for key in positions}
    for node in nodes:
        source = node['image_id']
        if source not in graph:
            continue
        for i, connected in enumerate(node['unobstructed']):
            target = nodes[i]['image_id']
            if connected and target in graph:
                graph[source][target] = float(np.linalg.norm(positions[source] - positions[target]))
    return graph


def graph_distances(graph, goal):
    distance = {goal: 0.0}
    queue = [(0.0, goal)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distance[node]:
            continue
        for target, weight in graph[node].items():
            candidate = cost + weight
            if candidate < distance.get(target, float('inf')):
                distance[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return distance


class MatterportNavigationEnv(NavigationEpisode, MetaEnv):
    """Actions: STOP=0 FORWARD=1 LEFT=2 RIGHT=3 UP=4 DOWN=5.

    FORWARD selects the closest angular navigable viewpoint in the current FOV,
    matching MatterSim index 1. Goal/teacher/path are never exposed to policy.
    """
    def __init__(self, config):
        import MatterSim
        self.config = config
        self.max_steps = int(getattr(config, 'max_timesteps', 100))
        self.success_distance = float(getattr(config, 'success_distance', 3.0))
        opener = gzip.open if str(config.episodes).endswith('.gz') else open
        with opener(config.episodes, 'rt', encoding='utf-8') as stream:
            raw = json.load(stream)
        self.episodes = []
        for item in raw:
            for i, instruction in enumerate(item['instructions']):
                self.episodes.append({**item, 'instruction': instruction,
                                      'episode_id': f"{item['path_id']}_{i}"})
        if not self.episodes:
            raise ValueError('R2R episode file is empty')
        self.index = int(getattr(config, 'rollout_index', 0))
        self.stride = int(getattr(config, 'episode_stride', 1))
        self.graphs = {}
        sim = MatterSim.Simulator()
        sim.setDatasetPath(str(Path(config.dataset_path).expanduser().resolve()))
        sim.setNavGraphPath(str(Path(config.connectivity_path).expanduser().resolve()))
        sim.setBatchSize(1)
        sim.setCameraResolution(getattr(config, 'image_size', 256), getattr(config, 'image_size', 256))
        sim.setCameraVFOV(np.deg2rad(60))
        sim.setDiscretizedViewingAngles(True)
        sim.setDepthEnabled(True)
        sim.initialize()
        super().__init__(sim)

    def reset(self):
        item = self.episodes[self.index % len(self.episodes)]
        self.index += self.stride
        self.episode = item
        self.episode_id = item['episode_id']
        scan = item['scan']
        if scan not in self.graphs:
            self.graphs[scan] = load_graph(Path(self.config.connectivity_path) / f'{scan}_connectivity.json')
        self.graph = self.graphs[scan]
        self.distances = graph_distances(self.graph, item['path'][-1])
        self.env.newEpisode([scan], [item['path'][0]], [item['heading']], [0.0])
        self._begin(self.distances[item['path'][0]])
        self.prev_obs = self.obs2meta(self.env.getState()[0])
        return self.prev_obs

    def obs2meta(self, state):
        # Heading/elevation and current visible degree are observable, unlike goal distance.
        features = np.asarray([state.heading, state.elevation, len(state.navigableLocations) - 1], dtype=np.float32)
        image = np.asarray(state.rgb)[..., ::-1].copy()  # MatterSim uses BGR.
        depth = np.asarray(state.depth, dtype=np.float32).squeeze(-1) if np.asarray(state.depth).ndim == 3 else np.asarray(state.depth, dtype=np.float32)
        return MetaObs(state=features, image=image[None].transpose(0, 3, 1, 2),
                       depth=depth[None] / 4000., raw_lang=self.episode['instruction'], timestep=self.steps)

    def meta2act(self, action):
        return discrete_action(action, 6)

    def step(self, action):
        from dataclasses import asdict
        if self.done:
            return asdict(self.prev_obs), 0., True, dict(self.last_info)
        action = self.meta2act(action)
        before = self.env.getState()[0]
        location, heading, elevation = 0, 0., 0.
        collision = False
        if action == 1:
            if len(before.navigableLocations) > 1:
                location = 1
            else:
                collision = True
        elif action in (2, 3):
            # Discretized MatterSim uses +/-1 increments, not radians.
            heading = -1. if action == 2 else 1.
        elif action in (4, 5):
            elevation = 1. if action == 4 else -1.
        if action != 0:
            self.env.makeAction([location], [heading], [elevation])
        after = self.env.getState()[0]
        source, target = before.location.viewpointId, after.location.viewpointId
        if source != target:
            self.travelled += self.graph[source][target]
        self.prev_obs = self.obs2meta(after)
        self.prev_obs.timestep = self.steps + 1
        return self._finish_step(self.distances.get(target, float('inf')), action == 0, collision)

    def close(self):
        # MatterSim's Python API has no close method; release renderer ownership.
        self.env = None
