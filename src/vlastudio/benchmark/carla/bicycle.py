"""Explicit opt-in repairs for disconnected bicycle lanes in public Town12."""
import math


class Waypoint:
    def __init__(self, original, network, *, transform=None, bridge_index=None):
        self.original, self.network = original, network
        self.transform = transform if transform is not None else original.transform
        self.bridge_index = bridge_index

    def __getattr__(self, name):
        return getattr(self.original, name)

    @property
    def is_junction(self):
        return True if self.bridge_index is not None else self.original.is_junction

    def next(self, distance):
        if not math.isclose(distance, 2.):
            raise ValueError('The bicycle repair supports the upstream two-meter plan only')
        if self.bridge_index is not None:
            return [self.network.bridge[self.bridge_index+1]]
        following = self.original.next(distance)
        if following:
            return [Waypoint(value, self.network) for value in following]
        if self.network.bridge is not None:
            return []
        return self.network.connect(self.original)


class LaneConnection:
    def __init__(self, map_instance, endpoint, *, max_gap=60.):
        self.map, self.endpoint, self.max_gap = map_instance, endpoint, max_gap
        self.bridge = None
        self.report = {}

    def connect(self, start):
        import carla
        endpoint = self.map.get_waypoint(self.endpoint, lane_type=carla.LaneType.Biking)
        if endpoint is None or endpoint.transform.location.distance(self.endpoint) > 2.:
            raise ValueError('The configured bicycle endpoint does not identify a nearby bicycle lane')
        end = endpoint
        visited = set()
        for _ in range(100):
            identity = (end.road_id, end.lane_id, round(end.s, 2))
            if identity in visited:
                raise ValueError('Cycle while tracing the configured bicycle endpoint backwards')
            visited.add(identity)
            previous = end.previous(2.)
            if not previous:
                break
            if len(previous) != 1:
                raise ValueError('Ambiguous bicycle endpoint predecessor')
            end = previous[0]
        else:
            raise ValueError('No nearby disconnected bicycle lane endpoint')
        a, z = start.transform.location, end.transform.location
        distance = a.distance(z)
        if not 2. < distance <= self.max_gap:
            raise ValueError(f'Bicycle lane gap is outside the declared bound: {distance:.2f}m')
        yaw = math.degrees(math.atan2(z.y-a.y, z.x-a.x))
        heading_error = lambda value: abs((value-yaw+180.) % 360.-180.)
        if max(heading_error(start.transform.rotation.yaw), heading_error(end.transform.rotation.yaw)) > 20.:
            raise ValueError('Configured bicycle endpoints do not continue in the same direction')
        # Only bridge the missing junction connection, bounded by native bicycle
        # lane endpoints selected from the original XML's start/end flow locations.
        count = math.ceil(distance/2.)
        nodes = []
        for index in range(1, count):
            fraction = index/count
            location = carla.Location(x=a.x+(z.x-a.x)*fraction, y=a.y+(z.y-a.y)*fraction,
                                      z=a.z+(z.z-a.z)*fraction)
            transform = carla.Transform(location, carla.Rotation(yaw=yaw))
            nodes.append(Waypoint(start, self, transform=transform, bridge_index=len(nodes)))
        nodes.append(Waypoint(end, self))
        self.bridge = nodes
        self.report = {'gap_meters': distance, 'interpolated_waypoints': len(nodes)-1,
                       'start': [a.x, a.y, a.z], 'end': [z.x, z.y, z.z]}
        return nodes[:1]

    def get_waypoint(self, location, *args, **kwargs):
        return Waypoint(self.map.get_waypoint(location, *args, **kwargs), self)

    def __getattr__(self, name):
        return getattr(self.map, name)


def repair_initializer(original, reports):
    def initialize(scenario, config):
        try:
            return original(scenario, config)
        except ValueError as error:
            if str(error) != "Couldn't find a proper plan for the bicycle flow":
                raise
            if 'end_actor_flow' not in config.other_parameters:
                raise ValueError('The bicycle repair requires an explicit end_actor_flow in the original XML') from error
        import carla
        values = config.other_parameters['end_actor_flow']
        endpoint = carla.Location(**{key: float(values[key]) for key in ['x', 'y', 'z']})
        original_map = scenario._map
        connection = LaneConnection(original_map, endpoint)
        scenario._map = connection
        try:
            result = original(scenario, config)
        finally:
            scenario._map = original_map
        if not connection.report:
            raise RuntimeError('No bicycle lane repair was actually applied')
        reports[config.name] = connection.report
        return result
    return initialize


def nonmutating_spawn(original):
    def spawn(flow, *args, **kwargs):
        # Upstream aliases the master plan and pops entries while setting up each
        # actor. Give it an actor-local copy so the declared continuous flow lasts.
        master = flow._plan
        flow._plan = list(master)
        try:
            return original(flow, *args, **kwargs)
        finally:
            flow._plan = master
    return spawn


def install(reports):
    import importlib
    import os
    from pathlib import Path
    import sys
    # RouteScenario discovers scenarios by their bare module names, rather than
    # their srunner package names. Patch that exact loaded class.
    directory = Path(os.environ['SCENARIO_RUNNER_ROOT'])/'srunner/scenarios'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    module = importlib.import_module('cross_bicycle_flow')
    module.CrossingBicycleFlow._initialize_actors = repair_initializer(
        module.CrossingBicycleFlow._initialize_actors, reports)
    class ContinuousBicycleFlow(module.BicycleFlow):
        _spawn_actor = nonmutating_spawn(module.BicycleFlow._spawn_actor)
    module.BicycleFlow = ContinuousBicycleFlow
    return module
