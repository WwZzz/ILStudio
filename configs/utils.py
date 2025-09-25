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


