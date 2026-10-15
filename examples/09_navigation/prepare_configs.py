"""Write train/val_unseen navigation configs with user-supplied asset paths.

Run from the repository root. This prepares YAML only; it does not import a
simulator, install dependencies, or download datasets.
"""
import argparse
from pathlib import Path

import yaml


def absolute(path):
    return Path(path).expanduser().resolve().as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vlnce-data', required=True, help='R2R_VLNCE_v1-3_preprocessed directory')
    parser.add_argument('--scenes-dir', required=True, help='Scene root containing mp3d/<scan>/<scan>.glb')
    parser.add_argument('--r2r-data', required=True, help='Directory containing R2R_train.json and R2R_val_unseen.json')
    parser.add_argument('--mattersim-prefix', required=True, help='Prefix passed to install_mattersim.py')
    parser.add_argument('--matterport-scans', default='data/v1/scans', help='Skybox root; unused when paper rendering is disabled')
    parser.add_argument('--output', default='configs/example09')
    args = parser.parse_args()
    templates = Path(__file__).resolve().parents[2]/'src/vlastudio/configs/env'
    habitat = yaml.safe_load((templates/'habitat/habitat_vlnce_cma.yaml').read_text())
    matterport = yaml.safe_load((templates/'matterport3d/matterport_r2r_vlnbert.yaml').read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val_unseen']:
        cma = dict(habitat)
        # Keep the packaged sensor/action protocol; replace dataset fields only.
        cma['habitat_overrides'] = [
            value for value in habitat['habitat_overrides']
            if not value.startswith(('habitat.dataset.data_path=',
                                     'habitat.dataset.scenes_dir=', 'habitat.dataset.split='))
        ] + [f'habitat.dataset.data_path={absolute(args.vlnce_data)}/{split}/{split}.json.gz',
             f'habitat.dataset.scenes_dir={absolute(args.scenes_dir)}',
             f'habitat.dataset.split={split}']
        r2r = dict(matterport, episodes=f'{absolute(args.r2r_data)}/R2R_{split}.json',
                   dataset_path=absolute(args.matterport_scans),
                   connectivity_path=f'{absolute(args.mattersim_prefix)}/source/connectivity')
        for name, config in [('cma', cma), ('r2r', r2r)]:
            path = output/f'{name}-{split.replace("_", "-")}.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            print(path)


if __name__ == '__main__':
    main()
