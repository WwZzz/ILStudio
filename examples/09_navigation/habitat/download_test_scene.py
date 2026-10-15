"""Fetch public Habitat test assets without requiring Git LFS."""
import argparse
import hashlib
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['skokloster-castle', 'apartment_1', 'van-gogh-room'],
                        default='skokloster-castle')
    parser.add_argument('--output', default='data/scene_datasets/habitat-test-scenes')
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    for suffix in ['glb', 'navmesh']:
        path = target / f'{args.scene}.{suffix}'
        if not path.exists():
            url = f'https://huggingface.co/datasets/ai-habitat/habitat_test_scenes/resolve/main/{path.name}'
            temporary = path.with_suffix(path.suffix + '.part')
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if suffix == 'glb':
                with temporary.open('rb') as stream:
                    if stream.read(4) != b'glTF':
                        raise RuntimeError(f'Download is not a binary glTF scene: {url}')
            temporary.replace(path)
        print(path, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
