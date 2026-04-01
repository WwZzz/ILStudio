import importlib
import multiprocessing as mp
import time
from typing import Any, List, Tuple

import yaml
from loguru import logger


def finalize_deploy_device_dict(item: dict) -> dict:
    """Normalize one deploy YAML entry (robot/teleop/viz row) and flatten ``args`` like ConfigLoader.load_robot."""
    from configs.loader import ConfigLoader

    if not isinstance(item, dict):
        return item
    cfg = ConfigLoader.normalize_config(item)
    if "args" in cfg and isinstance(cfg["args"], dict):
        for key, value in cfg["args"].items():
            if key not in cfg:
                cfg[key] = value
    if "type" in cfg:
        cfg["target"] = cfg["type"]
    return cfg


def coerce_finalize_deploy_list(cfg_or_list: Any) -> List:
    """If YAML root is a list, finalize each dict; if a single dict, wrap as one-element list."""
    if isinstance(cfg_or_list, list):
        return [finalize_deploy_device_dict(x) if isinstance(x, dict) else x for x in cfg_or_list]
    if isinstance(cfg_or_list, dict):
        return [finalize_deploy_device_dict(cfg_or_list)]
    return []


def is_visualizer_config(cfg: dict) -> bool:
    """True if ``type`` resolves to a class that subclasses ``BaseVisualizer`` (and is not the base itself)."""
    if not isinstance(cfg, dict):
        return False
    t = cfg.get("type")
    if not isinstance(t, str):
        return False
    t = t.strip()
    if "." not in t:
        return False
    mod_name, _, attr_name = t.rpartition(".")
    if not mod_name or not attr_name:
        return False
    try:
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr_name, None)
    except Exception:
        return False
    if not isinstance(obj, type):
        return False
    from deploy.visualizer.base import BaseVisualizer

    try:
        return issubclass(obj, BaseVisualizer) and obj is not BaseVisualizer
    except TypeError:
        return False


def partition_visualizer_configs(
    configs: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """Split ``configs`` into (device entries, visualizer entries)."""
    devices: List[dict] = []
    viz: List[dict] = []
    for c in configs:
        if is_visualizer_config(c):
            viz.append(c)
        else:
            devices.append(c)
    return devices, viz


def load_config(path: str) -> List[dict]:
    """Load config from yaml. Returns list of device configs."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, list):
        return raw
    return [raw]


def _device_spawns_children(cfg: dict) -> bool:
    """True if this device type spawns child processes (e.g. GUI), so it must not run in a daemon process."""
    device_type = cfg.get("type", "")
    return "keyboard" in device_type or "tk_slider" in device_type


def start_devices(configs: List[dict], daemon: bool = True) -> List[mp.Process]:
    """Start each device config in a separate process.
    
    Args:
        configs: List of device configurations
        daemon: If True, processes will be terminated when main process exits
                (including crashes/segfaults). Default True for safety.
                Devices that spawn children (e.g. Keyboard, TkSlider GUI) are always
                started with daemon=False so they can create subprocesses.
    """
    from deploy.base import start_device

    procs = []
    for cfg in configs:
        p = mp.Process(target=start_device, args=(cfg,))
        # Daemon processes cannot have children; Keyboard/TkSlider start GUI subprocesses
        p.daemon = False if _device_spawns_children(cfg) else daemon
        p.start()
        procs.append(p)
    return procs


def get_shm_name(cfg: dict) -> str:
    """Get SHM name from device config."""
    args = cfg.get("args", {})
    return args.get("name") or args.get("robot_id") or cfg.get("name", "unknown")


def load_device_configs(args, unknown_args) -> Tuple[List[dict], List[str], List[dict], List[dict]]:
    """
    Load device configs from args.

    Visualizer rows: ``type`` is a class subclassing ``BaseVisualizer`` (other than
    ``BaseVisualizer`` itself). Those rows are removed from the device list.

    Returns:
        Tuple of (all_device_configs, all_shm_names, teleop_device_configs, visualizer_configs)
    """
    from configs.loader import ConfigLoader

    from deploy.base import is_robot_config

    cfg_loader = ConfigLoader(args=args, unknown_args=unknown_args)
    robot_configs: List[dict] = []
    teleop_configs: List[dict] = []

    # Load robot config
    try:
        robot_cfg, robot_path = cfg_loader.load_robot(args.robot)
        robot_configs = coerce_finalize_deploy_list(robot_cfg)
        logger.info("Loaded robot config from: {}", robot_path)
    except FileNotFoundError as e:
        logger.warning("ConfigLoader failed, falling back to direct YAML: {}", e)
        robot_configs = coerce_finalize_deploy_list(load_config(args.robot))

    # Load teleop config
    if args.teleop.strip():
        try:
            teleop_cfg, teleop_path = cfg_loader.load_teleop(args.teleop)
            teleop_configs = coerce_finalize_deploy_list(teleop_cfg)
            logger.info("Loaded teleop config from: {}", teleop_path)
        except FileNotFoundError as e:
            logger.warning("ConfigLoader failed, falling back to direct YAML: {}", e)
            teleop_configs = coerce_finalize_deploy_list(load_config(args.teleop))

    robot_devices, robot_viz = partition_visualizer_configs(robot_configs)
    teleop_devices, teleop_viz = partition_visualizer_configs(teleop_configs)
    visualizer_configs = teleop_viz + robot_viz

    # Auto-fill control_shm_name for robot configs if teleop is specified
    if teleop_devices:
        teleop_shm_name = get_shm_name(teleop_devices[0])
        for cfg in robot_devices:
            try:
                if is_robot_config(cfg):
                    args_dict = cfg.get("args", {})
                    if args_dict.get("control_shm_name") is None:
                        if "args" not in cfg:
                            cfg["args"] = {}
                        cfg["args"]["control_shm_name"] = teleop_shm_name
                        logger.info(
                            "Auto-set control_shm_name='{}' for robot '{}'",
                            teleop_shm_name,
                            get_shm_name(cfg),
                        )
            except Exception:
                pass
    if teleop_devices:
        logger.info("Teleop devices: {}", [get_shm_name(c) for c in teleop_devices])
    logger.info("Robot devices: {}", [get_shm_name(c) for c in robot_devices])
    if visualizer_configs:
        logger.info(
            "Visualizer entries: {}",
            [v.get("name") or v.get("type") for v in visualizer_configs],
        )
    all_configs = teleop_devices + robot_devices
    all_shm_names = [get_shm_name(c) for c in all_configs]
    return all_configs, all_shm_names, teleop_devices, visualizer_configs


class RateLimiter:
    """
    A class to manage the rate for a single thread.
    Each thread should have its own instance.
    """

    def __init__(self):
        self._last_sleep_time = time.perf_counter()

    def sleep(self, rate: float):
        """
        Sleeps for a duration that maintains the desired loop rate.

        Args:
            rate (float): The desired loop frequency in Hz.
        """
        if rate <= 0:
            return

        target_period = 1.0 / rate
        current_time = time.perf_counter()
        elapsed_time = current_time - self._last_sleep_time
        sleep_duration = target_period - elapsed_time

        if sleep_duration > 0:
            time.sleep(sleep_duration)

        # Update the timestamp for the next iteration
        self._last_sleep_time = time.perf_counter()