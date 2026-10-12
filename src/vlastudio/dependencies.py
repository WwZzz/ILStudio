"""Dependency guidance using installed metadata, without importing ML packages."""
import importlib.metadata as metadata
import os
import shlex
import sys


class DependencyError(ImportError):
    """A component's declared runtime is missing or incompatible."""
    def __init__(self, plan):
        self.plan = plan
        super().__init__(format_plan(plan))


def inspect_environment(profile):
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    from .profiles import _requirements, platform_issues
    issues = platform_issues(profile)
    python = profile['python']
    spec = profile.get('requires_python') or (python if python.startswith(('>', '<', '=', '!', '~')) else '==' + python + '.*')
    if Version('.'.join(map(str, sys.version_info[:3]))) not in SpecifierSet(spec):
        issues.append(f"Requires Python {spec}; current Python is {sys.version.split()[0]}")
    requirements = profile.get('requirements', [])
    if profile.get('lock'):
        requirements = [line.strip() for line in profile['lock'].splitlines()
                        if line.strip() and not line.lstrip().startswith('#')]
    requirements = _requirements(requirements, profile.get('overrides', []))
    packages = []
    for value in requirements:
        req = Requirement(value)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            version = None
        status = 'missing' if version is None else ('version mismatch' if req.specifier and version not in req.specifier else 'installed')
        packages.append({'requirement': value, 'installed': version, 'status': status})
        if status != 'installed':
            issues.append(f'{value}: {status}' + (f' (installed {version})' if version else ''))
    return issues, packages


def _command(tokens):
    if os.name != 'nt':
        return shlex.join(map(str, tokens))
    # PowerShell literal strings, including paths containing spaces or apostrophes.
    def quote(value):
        value = str(value)
        return value if all(c.isalnum() or c in '/\\._:-=' for c in value) else "'" + value.replace("'", "''") + "'"
    return ' '.join(quote(token) for token in tokens)


def environment_plan(*, policy=None, environment=None, dataset=None, manifest=None, remote=False):
    """Read-only installation plan for the calling interpreter."""
    from .env_cli import select_profile
    kind, profile = select_profile(policy, environment, manifest, remote, dataset, check_platform=False)
    issues, packages = inspect_environment(profile)
    selectors = []
    for key, value in (('policy', policy), ('environment', environment), ('dataset', dataset), ('manifest', manifest)):
        if value:
            selectors.append(f'{key}={str(value)!r}')
    if remote:
        selectors.append('remote=True')
    arguments = ', '.join(selectors)
    setup = f'import vlastudio as vla; vla.install_dependencies({arguments})'
    return {'environment': kind, 'python': sys.executable, 'required_python': profile['python'],
            'profile': profile, 'issues': issues, 'packages': packages,
            'commands': {
                'create': _command(['python', '-m', 'uv', 'venv', '--python', profile['python'], '--seed', '.venv']),
                'install_current': _command(['python', '-c', setup]),
                'run': 'python run.py',
                'check': _command(['python', '-c', f'import vlastudio as vla; print(vla.environment_plan({arguments}))']),
            }, 'setup_python': 'import vlastudio as vla\n' + f'vla.install_dependencies({arguments})\n'}


def install_dependencies(*, policy=None, environment=None, dataset=None, manifest=None,
                         remote=False, cache_dir=None, index_url=None, offline=False,
                         dry_run=False):
    """Explicitly install a component recipe into the calling Python interpreter.

    Call from a setup script in your own virtual environment. Uses uv's additive
    installer and the same inherited recipes as dependency checks. Importing
    VLAStudio and loading components never invoke this function automatically.
    dry_run returns a read-only plan without creating caches or installing.
    """
    plan = environment_plan(policy=policy, environment=environment, dataset=dataset,
                            manifest=manifest, remote=remote)
    if dry_run:
        return plan
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    from .env_cli import select_profile, installation_env
    from .env_registry import install
    from .paths import cache_root
    _, profile = select_profile(policy, environment, manifest, remote, dataset)
    python = profile['python']
    spec = profile.get('requires_python') or (python if python.startswith(('>', '<', '=', '!', '~')) else '==' + python + '.*')
    if Version('.'.join(map(str, sys.version_info[:3]))) not in SpecifierSet(spec):
        raise ValueError(f'This recipe requires Python {spec}; create and activate that Python environment before running your setup script. Current interpreter: {sys.executable}')
    cache = cache_root(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    install(profile, sys.executable, cache, installation_env(cache, index_url, offline))
    return environment_plan(policy=policy, environment=environment, dataset=dataset,
                            manifest=manifest, remote=remote)


def format_plan(plan):
    lines = [f"VLAStudio runtime: {plan['environment']} (Python {plan['required_python']})",
             f"Current interpreter: {plan['python']}"]
    if plan['issues']:
        lines.append('Missing or incompatible dependencies:')
        lines.extend('  - ' + issue for issue in plan['issues'])
    else:
        lines.append('Declared package versions are installed (metadata check only).')
    lines.extend(['Create and activate a compatible virtual environment:',
                  '  ' + plan['commands']['create'],
                  '  Activate .venv, then install VLAStudio there from the same source checkout or wheel.',
                  'Save the following as setup_environment.py and run python setup_environment.py:',
                  *('  ' + line for line in plan['setup_python'].splitlines()),
                  'Then run your own workflow:', '  ' + plan['commands']['run'],
                  'Dependency installation is explicit and may change package versions in the selected interpreter.'])
    lines.extend('Note: ' + note for note in plan['profile'].get('notes', []))
    lines.append('Checks do not verify CUDA, native libraries, assets, import health, extras or VCS revisions.')
    return '\n'.join(lines)


def require_dependencies(**selectors):
    plan = environment_plan(**selectors)
    if plan['issues']:
        raise DependencyError(plan)
    return plan
