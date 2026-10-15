"""Unpack a small, deterministic scene subset from an already mounted HM3D tar.

No download or asset-access agreement is performed. Keep the original archive
mounted; only geometry and navigation meshes for the selected scenes are written.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scene-count', type=int, default=2)
    args = parser.parse_args()
    if args.scene_count < 1:
        parser.error('--scene-count must be positive')
    destination = Path(args.output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        scenes = sorted(m.name for m in members if m.isfile() and m.name.endswith('.basis.glb'))
        if len(scenes) < args.scene_count:
            raise ValueError(f'Archive contains only {len(scenes)} scene geometries')
        scenes = scenes[:args.scene_count]
        wanted = set(scenes) | {s.removesuffix('.glb') + '.navmesh' for s in scenes}
        for member in members:
            if member.name not in wanted:
                continue
            relative = PurePosixPath(member.name)
            target = (destination / relative).resolve()
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or not target.is_relative_to(destination)):
                raise ValueError(f'Unsafe scene member: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
            extracted.append({'member': member.name, 'bytes': member.size})
    manifest = {'archive': str(Path(args.archive).expanduser().resolve()),
                'selection': 'First lexicographically sorted .basis.glb scenes; fixed before training',
                'scenes': [str(destination / s) for s in scenes], 'extracted': extracted,
                'scene_sha256': {s: hashlib.sha256((destination / s).read_bytes()).hexdigest() for s in scenes}}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
