import os
from pathlib import Path

def resolve_yaml(name_or_path: str, base_dir: str) -> str:
    """Resolve a YAML config by either absolute/relative path or by name within base_dir.
    Returns an existing file path or raises FileNotFoundError.
    """
    if not name_or_path:
        raise FileNotFoundError("Empty config name or path")

    p = Path(name_or_path)
    # If looks like a path and exists, return as-is
    if p.suffix.lower() == '.yaml' or p.suffix.lower() == '.yml' or any(sep in name_or_path for sep in ['/', '\\']):
        candidate = Path(name_or_path)
        if candidate.exists():
            return str(candidate)
        # If it's a path without extension, try adding .yaml
        if candidate.suffix == '' and candidate.with_suffix('.yaml').exists():
            return str(candidate.with_suffix('.yaml'))
        raise FileNotFoundError(f"Config not found: {name_or_path}")

    # Treat as name under base_dir
    base = Path(base_dir)
    candidate = base / f"{name_or_path}.yaml"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Config '{name_or_path}' not found under {base_dir}")


def _set_nested(obj, keys, value):
    cur = obj
    for k in keys[:-1]:
        if isinstance(cur, dict):
            if k not in cur or not isinstance(cur[k], (dict,)):
                cur[k] = {}
            cur = cur[k]
        else:
            if not hasattr(cur, k) or not isinstance(getattr(cur, k), (dict,)):
                setattr(cur, k, {})
            cur = getattr(cur, k)
    last = keys[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def parse_overrides(unknown_args):
    overrides = { 'task': {}, 'training': {}, 'policy': {}, 'teleop': {}, 'robot': {}, 'env': {} }
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith('--'):
            i += 1
            continue
        key = token[2:]
        value = None
        if '=' in key:
            key, value = key.split('=', 1)
        else:
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith('--'):
                value = unknown_args[i+1]
                i += 1
        roots = ('task.', 'training.', 'policy.', 'teleop.', 'robot.', 'env.')
        if key.startswith(roots):
            root, subpath = key.split('.', 1)
            overrides[root][subpath] = value
        i += 1
    return overrides


def apply_overrides_to_mapping(mapping_obj, flat_overrides, caster):
    for dotted, raw in flat_overrides.items():
        if raw is None:
            continue
        try:
            val = caster(raw)
        except Exception:
            val = raw
        keys = dotted.split('.')
        _set_nested(mapping_obj, keys, val)


def apply_overrides_to_object(obj, flat_overrides, caster):
    for dotted, raw in flat_overrides.items():
        if raw is None:
            continue
        try:
            val = caster(raw)
        except Exception:
            val = raw
        keys = dotted.split('.')
        _set_nested(obj, keys, val)


