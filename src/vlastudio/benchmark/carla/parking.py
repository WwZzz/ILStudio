"""Avoid duplicate stock/benchmark parking and keep each route independent."""
import copy


class ParkingCatalog:
    def __init__(self, module):
        self.module = module
        self.original = {name: copy.deepcopy(value) for name, value in vars(module).items()
                         if name.startswith('Town') and isinstance(value, list)}

    def reset(self, town):
        # Upstream removes entries in place while filtering/spawning. Its usual
        # one-route-per-process launcher starts with a fresh catalog each time.
        if town in self.original:
            setattr(self.module, town, copy.deepcopy(self.original[town]))


def configure_parking(world, remove_map_parked_vehicles=True, *, labels=None):
    if labels is None:
        import carla
        labels = [carla.CityObjectLabel.Car, carla.CityObjectLabel.Truck, carla.CityObjectLabel.Bus]
    identifiers = set()
    if remove_map_parked_vehicles:
        for label in labels:
            identifiers.update(value.id for value in world.get_environment_objects(label)
                               if value.name.startswith('BP_ParkCars'))
        if identifiers:
            world.enable_environment_objects(identifiers, False)
            world.tick()
    return {'remove_map_parked_vehicles': remove_map_parked_vehicles,
            'disabled_stock_parked_objects': len(identifiers), 'parking_catalog_reset': True}


def audited_builder(original, trace):
    """Observe the unmodified builder and reject swallowed initialization errors."""
    def build(scenario, *args, **kwargs):
        result = original(scenario, *args, **kwargs)
        built = {value.config.name for value in scenario.list_scenarios}
        pending = {value.name for value in scenario.missing_scenario_configurations}
        requested = {value.name for value in scenario.scenario_configurations}
        trace.setdefault('initialized', set()).update(built)
        trace.setdefault('actors', {}).update({value.config.name: len(value.other_actors)
                                             for value in scenario.list_scenarios})
        trace['background_parked_objects'] = len(scenario._parked_ids)
        missing = requested-built-pending
        if missing:
            trace.setdefault('errors', []).extend(sorted(missing))
            raise RuntimeError('Official scenario initialization failed: '+', '.join(sorted(missing)))
        return result
    return build
