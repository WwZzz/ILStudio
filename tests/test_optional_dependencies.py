import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest

import vlastudio
from vlastudio import optional_dependencies as optional


ROOT = Path(__file__).parents[1]


def test_import_vlastudio_does_not_import_heavy_stacks():
    script = (
        "import sys, vlastudio; "
        "blocked={'torch','torchvision','transformers','diffusers','tensorflow','jax'}; "
        "loaded=sorted(blocked.intersection(sys.modules)); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_dataset_error_names_shared_layers_and_commands(monkeypatch):
    monkeypatch.setattr(optional, "find_spec", lambda name: None)
    with pytest.raises(vlastudio.OptionalDependencyError) as caught:
        vlastudio.load_dataset("sim_transfer_cube_scripted")
    message = str(caught.value)
    assert "numpy" in message and "torch" in message
    assert 'vlastudio[data,torch]' in message
    assert '-e ".[data,torch]"' in message


def test_fastwam_error_selects_only_fastwam_extra(monkeypatch):
    monkeypatch.setattr(optional, "find_spec", lambda name: None)
    with pytest.raises(vlastudio.OptionalDependencyError) as caught:
        vlastudio.load_policy("fastwam")
    message = str(caught.value)
    assert "fastwam" in message
    assert 'vlastudio[fastwam]' in message
    assert "data,torch,fastwam" not in message


def test_act_error_selects_act_extra(monkeypatch):
    monkeypatch.setattr(optional, "find_spec", lambda name: None)
    with pytest.raises(vlastudio.OptionalDependencyError) as caught:
        vlastudio.load_policy("act")
    assert 'vlastudio[act]' in str(caught.value)


def test_dependency_check_can_be_delegated_to_managed_worker(monkeypatch):
    monkeypatch.setattr(optional, "find_spec", lambda name: None)
    dataset = vlastudio.load_dataset("sim_transfer_cube_scripted", check_dependencies=False)
    policy = vlastudio.load_policy("fastwam", check_dependencies=False)
    assert dataset.config_path.is_file()
    assert policy.config_path.is_file()


def test_declared_extras_cover_every_diagnostic_group():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]
    assert {group.extra for group in optional.GROUPS.values()} <= set(extras)
    assert {"data", "torch", "base"} <= set(extras)


def test_every_builtin_policy_config_has_a_dependency_route():
    policy_dir = ROOT / "src" / "vlastudio" / "configs" / "policy"
    components = {optional.policy_component(path) for path in policy_dir.glob("*.yaml")}
    assert None not in components
    assert components <= set(optional.POLICY_GROUPS)
