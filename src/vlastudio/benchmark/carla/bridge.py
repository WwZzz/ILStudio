"""Step the official Bench2Drive scenario manager through a private process pipe."""
import contextlib
import os
from pathlib import Path
import sys
import signal
from types import SimpleNamespace

AGENT_PIPE = None


def prepare_sources(carla_root):
    from vlastudio.utils.driving_assets import source
    b2d, tcp = source('bench2drive'), source('tcp')
    for path in [b2d/'leaderboard', b2d/'scenario_runner', tcp, Path(carla_root)/'PythonAPI/carla']:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ['IS_BENCH2DRIVE'] = 'True'
    os.environ['PLANNER_TYPE'] = 'only_traj'
    os.environ['SCENARIO_RUNNER_ROOT'] = str(b2d/'scenario_runner')
    os.environ['LEADERBOARD_ROOT'] = str(b2d/'leaderboard')
    os.environ['CARLA_ROOT'] = str(carla_root)
    os.environ['CHALLENGE_TRACK_CODENAME'] = 'SENSORS'
    os.environ.pop('SAVE_PATH', None)
    # Upstream targets Python 3.7/3.8. Restrict this API compatibility adapter
    # to its XML readers; scenario trees and parameters are otherwise unchanged.
    import importlib
    import xml.etree.ElementTree as ET
    class Element(ET.Element):
        def getchildren(self):
            return list(self)
    def parse(filename):
        return ET.parse(filename, parser=ET.XMLParser(target=ET.TreeBuilder(element_factory=Element)))
    for name in ['leaderboard.utils.route_parser', 'srunner.tools.scenario_parser', 'srunner.tools.route_parser']:
        module = importlib.import_module(name)
        module.ET = SimpleNamespace(parse=parse)
    return b2d


def worker(pipe, config):
    global AGENT_PIPE
    AGENT_PIPE = pipe
    def terminate(signum, frame):
        raise SystemExit('Driving worker terminated')
    signal.signal(signal.SIGTERM, terminate)
    server = evaluator = client = None
    work = Path(config['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    try:
        with (work/'scenario.log').open('a', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from .server import CarlaServer, free_ports
            server = CarlaServer(config.get('carla_root'), gpu=config.get('gpu', 0),
                graphics_runtime=config.get('graphics_runtime'), run_dir=work/'server',
                timeout=config.get('startup_timeout', 180.))
            client = server.connect()
            b2d = prepare_sources(server.root)
            os.chdir(b2d)
            import carla
            from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
            from leaderboard.utils.statistics_manager import StatisticsManager
            from leaderboard.utils.route_indexer import RouteIndexer

            class OwnedEvaluator(LeaderboardEvaluator):
                def _setup_simulation(self, args):
                    # Own the server lifecycle; upstream's broad crash-kill path
                    # is never executed by this adapter.
                    self.carla_path = str(server.root)
                    args.port = server.port
                    client.set_timeout(args.timeout)
                    settings = carla.WorldSettings(synchronous_mode=True,
                        fixed_delta_seconds=.05, deterministic_ragdolls=True,
                        spectator_as_ego=False)
                    client.get_world().apply_settings(settings)
                    args.traffic_manager_port = free_ports()
                    traffic = client.get_trafficmanager(args.traffic_manager_port)
                    traffic.set_synchronous_mode(True)
                    traffic.set_hybrid_physics_mode(True)
                    return client, args.timeout, traffic

            args = SimpleNamespace(host='127.0.0.1', port=server.port,
                traffic_manager_port=0, traffic_manager_seed=config.get('seed', 0),
                debug=0, record='', timeout=config.get('step_timeout', 600.),
                routes=config['routes'], routes_subset='', repetitions=1,
                agent=str(Path(__file__).with_name('vlastudio_b2d_agent.py')),
                agent_config='vlastudio', track='SENSORS', resume=False,
                checkpoint=str(work/'official-results.json'),
                debug_checkpoint=str(work/'official-live.txt'), gpu_rank=config.get('gpu', 0))
            statistics = StatisticsManager(args.checkpoint, args.debug_checkpoint)
            evaluator = OwnedEvaluator(args, statistics)
            indexer = RouteIndexer(args.routes, 1, '')
            statistics.clear_records()
            statistics.save_progress(0, indexer.total)
            statistics.write_statistics()
            while indexer.peek():
                route = indexer.get_next_config()
                log.flush()
                log_start = log.tell()
                crashed = evaluator._load_and_run_scenario(args, route)
                log.flush()
                with (work/'scenario.log').open('rb') as recorded:
                    recorded.seek(log_start)
                    text = recorded.read().decode(errors='replace')
                if 'Skipping scenario ' in text:
                    raise RuntimeError('Official scenario setup failed; this route cannot be reported as a valid benchmark run')
                statistics.save_progress(indexer.index, indexer.total)
                statistics.write_statistics()
                records = statistics._results.to_json()['_checkpoint']['records']
                if not records:
                    raise RuntimeError('Official evaluator did not produce a route record')
                if not indexer.peek() and not crashed:
                    statistics.compute_global_statistics()
                    statistics.validate_and_write_statistics(evaluator.sensors_initialized, False)
                pipe.send(('done', records[-1]))
                if not indexer.peek():
                    break
                command, _ = pipe.recv()
                if command == 'close':
                    break
                if command != 'reset':
                    raise ValueError('Expected reset after terminal route')
                if crashed:
                    raise RuntimeError('CARLA reported a simulator crash; the failed route record is preserved')
            else:
                evaluator._reset_world_settings()
                statistics.compute_global_statistics()
                statistics.validate_and_write_statistics(evaluator.sensors_initialized, False)
                pipe.send(('complete', None))
    except BaseException:
        import traceback
        try:
            pipe.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        try:
            if client is not None:
                client.set_timeout(2.)
            if evaluator is not None:
                evaluator._cleanup()
        except Exception:
            pass
        finally:
            if server is not None:
                server.close()
            pipe.close()
