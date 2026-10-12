"""Select reusable environments by component configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .configuration import read_config
from .paths import cache_root, runtime_env
from .profiles import environment_for
from .runtime import prepare, environment_target, snapshot

def current_python():
    """Keep installation in the selected environment and reject stale paths."""
    selected = os.environ.get('VLASTUDIO_ACTIVE_PYTHON')
    active = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX')
    if not selected and active:
        selected = str(Path(active) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
    python = Path(selected or sys.executable).expanduser().absolute()
    if not python.is_file():
        raise ValueError(
            f'Current environment interpreter does not exist: {python}. '
            'The environment may have been moved or removed. '
            'Repair its activation script and activate it again, or select an existing environment before installing.'
        )
    return python


def installation_env(cache, index_url=None, offline=False):
    env = runtime_env(cache)
    if index_url:
        env['UV_INDEX_URL'] = index_url
        env.pop('UV_DEFAULT_INDEX', None)
    if offline:
        env['UV_OFFLINE'] = '1'
    return env


def select_profile(policy=None, environment=None, manifest=None, remote=False,
                   dataset=None, *, check_platform=True):
    from .profiles import combine_profiles, dataset_environment, recipe
    if manifest:
        if policy or environment or dataset:
            raise ValueError('Use either a complete --runtime-manifest or component selectors')
        return 'custom', environment_for({}, Path.cwd() / 'config.yaml',
                                         str(Path(manifest).expanduser().resolve()),
                                         check_platform=check_platform)
    selected = []
    if policy and not remote:
        policy = {'openpi': 'pi0', 'dp': 'diffusion_policy'}.get(policy, policy)
        config, path = read_config(policy, 'policy')
        selected.append(('policy', environment_for(config, path, check_platform=check_platform)))
    elif remote:
        selected.append(('remote', environment_for({}, recipe('remote'), str(recipe('remote')),
                                                   check_platform=check_platform)))
    if environment:
        family = {'robotwin': 'robotwin', 'behavior': 'behavior1k', 'behavior1k': 'behavior1k'}.get(environment)
        if family:
            profile = environment_for({}, recipe(family), str(recipe(family)), check_platform=check_platform)
        else:
            config, path = read_config('aloha_transfer' if environment == 'aloha_sim' else environment, 'env')
            configs = config if isinstance(config, list) else [config]
            profiles = [environment_for(c, path, check_platform=check_platform) for c in configs]
            profile = profiles[0]
            for extra in profiles[1:]:
                profile = combine_profiles(profile, extra)
        selected.append(('env', profile))
    if dataset:
        config, path = read_config(dataset, 'task')
        # Shared data requirements work on the selected model's Python version.
        profile = dataset_environment(config, path, check_platform=check_platform,
                                      python=selected[0][1]['python'] if selected else None)
        selected.append(('data', profile))
    if not selected:
        return 'base', environment_for({}, recipe('base'), str(recipe('base')), check_platform=check_platform)
    kind, profile = selected[0]
    for _, extra in selected[1:]:
        profile = combine_profiles(profile, extra)
    return ('combined' if len(selected) > 1 else kind), profile


# Installation and creation use the same resolved dependency contract.
install_profile = select_profile


def main(argv):
    from . import env_registry
    parser = argparse.ArgumentParser(prog='vlastudio env')
    parser.add_argument('action', choices=['create', 'install', 'activate', 'deactivate', 'init', 'prepare', 'run', 'path', 'list', 'plan', 'check', 'catalog'])
    parser.add_argument('name', nargs='?')
    parser.add_argument('-n', '--name', dest='environment_name')
    parser.add_argument('--last-created', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--shell', choices=['bash', 'zsh', 'powershell'])
    parser.add_argument('--policy')
    parser.add_argument('--env')
    parser.add_argument('--dataset', nargs='?', const='sim_transfer_cube_scripted', help='Task config; omit the value for shared data loaders')
    parser.add_argument('--json', action='store_true', help='Machine-readable plan/check output')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--runtime-manifest')
    parser.add_argument('-i', '--index-url', help='Package index URL for this invocation; otherwise inherit system configuration')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if args.action in ('plan', 'check', 'catalog'):
        from .dependencies import environment_plan, format_plan
        try:
            if args.action == 'catalog':
                from .profiles import POLICY_PROFILES
                for component, family in POLICY_PROFILES.items():
                    if component == 'remote':
                        print('remote policy client: vlastudio env plan --remote --env <env-config>')
                        continue
                    print(f'policy {component} -> {family}; vlastudio env plan --policy ' +
                          ('pi0' if component == 'openpi' else component))
                print('data: vlastudio env plan --dataset <task-config>')
                print('simulators: vlastudio env plan --env <env-config>')
                print('Other policies: their local pyproject.toml is exposed as a legacy recipe.')
                return 0
            plan = environment_plan(policy=args.policy, environment=args.env,
                                    dataset=args.dataset, manifest=args.runtime_manifest, remote=args.remote)
            print(json.dumps(plan, indent=2) if args.json else format_plan(plan))
            return 2 if args.action == 'check' and plan['issues'] else 0
        except (ValueError, OSError) as error:
            print(f'vlastudio env: {error}', file=sys.stderr)
            return 2
    cache = cache_root(args.cache_dir)
    if not args.dry_run:
        try:
            env_registry.ensure_base(cache)
        except (ValueError, OSError) as error:
            print(f'vlastudio env: {error}', file=sys.stderr)
            return 2
    if args.action == 'init':
        print(env_registry.hook(args.shell or 'bash'))
        return 0
    if args.action in ('activate', 'deactivate'):
        try:
            if not args.shell:
                raise ValueError('Initialize the shell first: eval "$(vlastudio env init --shell bash)"; PowerShell: vlastudio env init --shell powershell | Out-String | Invoke-Expression')
            label = args.environment_name or args.name or 'base'
            if args.last_created:
                label = (cache / 'last-environment').read_text(encoding='utf-8')
            if args.action == 'deactivate':
                label = 'base'
            print(env_registry.activation(cache, label, args.shell))
            return 0
        except (ValueError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 2
    if args.action in ('create', 'prepare'):
        explicit_name = args.environment_name or args.name
        if explicit_name:
            python = env_registry.existing_python(cache, explicit_name)
            if python is not None:
                if args.dry_run:
                    print(json.dumps({'environment': explicit_name, 'python': str(python), 'existing': True}, indent=2))
                    return 0
                kind = env_registry.read(cache).get(explicit_name, {}).get('kind') or 'managed'
                label = env_registry.register(cache, explicit_name, python, kind)
                print(f'{label}: {python}')
                return 0
    if args.action == 'list':
        entries = env_registry.read(cache)
        for name, entry in sorted(entries.items()):
            state = 'ready' if Path(entry['python']).is_file() else 'missing'
            print(f"{name}\t{entry['kind']}\t{state}\t{entry['python']}")
        for ready in sorted((cache / 'envs').glob('*/.ready')):
            python = ready.parent / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            if not any(e['python'] == str(python) for e in entries.values()):
                print(f'{ready.parent.name}\tmanaged\t{python}')
        if os.environ.get('VLASTUDIO_BASE_PYTHON'):
            print('base: ' + os.environ['VLASTUDIO_BASE_PYTHON'])
        return 0
    try:
        selector = install_profile if args.action == 'install' else select_profile
        name, profile = selector(args.policy, args.env, args.runtime_manifest, args.remote, args.dataset)
        explicit_name = args.environment_name or args.name
        label = explicit_name or (name if name == 'base' else args.policy or args.env or ('data' if args.dataset else 'custom'))
        if args.action == 'install':
            label = explicit_name or 'base'
            registered = env_registry.read(cache).get(explicit_name, {}) if explicit_name else {}
            python = Path(registered['python']) if registered else current_python()
            if not python.is_file():
                raise ValueError(f'Environment interpreter does not exist: {python}')
            if args.dry_run:
                print(json.dumps({'python': str(python), 'profile': profile, 'index_url': args.index_url}, indent=2))
                return 0
            env = installation_env(cache, args.index_url, args.offline)
            env_registry.install(profile, python, cache, env)
            label = env_registry.register(cache, label, python, 'installed')
            print(f'{label}: {python}')
            return 0
        python = environment_target(profile, cache, explicit_name) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        registered = env_registry.read(cache).get(label, {})
        external = registered.get('python')
        if registered.get('kind') == 'default' and args.action in ('create', 'prepare'):
            external = None
        if name == 'base' and label == 'base':
            external = os.environ.get('VLASTUDIO_BASE_PYTHON') or external
        if external:
            python = Path(os.path.abspath(os.path.expanduser(external)))
            if not python.is_file():
                raise ValueError(f'Base interpreter does not exist: {python}')
        if args.dry_run:
            print(json.dumps({'environment': name, 'python': str(python), 'profile': profile, 'index_url': args.index_url}, indent=2))
            return 0
        if args.action == 'path':
            print(python)
            return 0
        env = installation_env(cache, args.index_url, args.offline)
        if not external:
            if explicit_name:
                python = prepare(profile, cache, env, args.offline, name=explicit_name)
            else:
                python = prepare(profile, cache, env, args.offline)
        label = env_registry.register(cache, label, python, 'external' if external else 'managed')
        if args.action in ('prepare', 'create'):
            print(f'{label}: {python}')
            return 0
        if not command:
            raise ValueError('Provide a command after --, for example -- python experiment.py')
        app = snapshot(cache)
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        env['VIRTUAL_ENV'] = str(python.parent.parent)
        env['PYTHONPATH'] = os.pathsep.join([str(app), os.getcwd(), env.get('PYTHONPATH', '')])
        if command[0] in ('python', 'python3'):
            command[0] = str(python)
        return subprocess.call(command, env=env)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'vlastudio env: {error}', file=sys.stderr)
        return 2
