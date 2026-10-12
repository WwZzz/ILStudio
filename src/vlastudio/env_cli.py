"""Read-only dependency profile selection and inspection."""
import argparse
import json
from pathlib import Path
import sys

from .configuration import read_config
from .profiles import environment_for


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


def main(argv):
    parser = argparse.ArgumentParser(prog="vlastudio env")
    parser.add_argument("action", choices=["plan", "check", "catalog"])
    parser.add_argument("--policy")
    parser.add_argument("--env")
    parser.add_argument("--dataset", nargs="?", const="sim_transfer_cube_scripted")
    parser.add_argument("--runtime-manifest")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from .dependencies import environment_plan, format_plan
    try:
        if args.action == "catalog":
            from .profiles import POLICY_PROFILES
            print(json.dumps(POLICY_PROFILES, indent=2))
            return 0
        plan = environment_plan(policy=args.policy, environment=args.env,
                                dataset=args.dataset, manifest=args.runtime_manifest, remote=args.remote)
        print(json.dumps(plan, indent=2) if args.json else format_plan(plan))
        return 2 if args.action == "check" and plan["issues"] else 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
