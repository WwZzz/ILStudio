"""Load bundled ALOHA assets, including checkouts in Unicode Windows paths."""
import os
from pathlib import Path

from dm_control import mujoco


def load_physics(xml_path):
    path = Path(xml_path)
    if os.name != "nt" or str(path).isascii():
        return mujoco.Physics.from_xml_path(str(path))
    # MuJoCo's native Windows file opener cannot read some non-ASCII paths.
    # Python reads these files correctly; MuJoCo can consume the bytes through VFS.
    assets = {p.relative_to(path.parent).as_posix(): p.read_bytes()
              for p in path.parent.rglob("*") if p.is_file()}
    return mujoco.Physics.from_xml_string(path.read_text(encoding="utf-8"), assets=assets)
