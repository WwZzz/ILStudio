"""Install a selected recipe into this Python environment, using the library API.

Activate your own virtual environment and install VLAStudio before running this
script. Workload scripts only import/use the library; installation is separate.
"""
import argparse
import json

import vlastudio as vla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy')
    parser.add_argument('--env', dest='environment')
    parser.add_argument('--dataset')
    parser.add_argument('--runtime-manifest', dest='manifest')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('-i', '--index-url')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    plan = vla.install_dependencies(**vars(options))
    if options.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        print('Dependencies installed. Run your workflow with this Python interpreter.')


if __name__ == '__main__':
    main()
