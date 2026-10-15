"""Build MatterSim for the current Python on Ubuntu 22.04 without system installs.

Requires apt repository metadata, a C++ compiler, pkg-config, CMake <4, requests,
NumPy and NVIDIA EGL drivers. All downloaded sources/libraries stay in --prefix.
The installer never downloads scans or accepts dataset terms. Source the emitted
activate.sh only in the process that will run MatterSim.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile

import requests

REVISION = '589d091b111333f9e9f9d6cfd021b2eb68435925'


def extract_source(url, target):
    response = requests.get(url, timeout=(20, 180))
    response.raise_for_status()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            member.name = str(Path(*parts))
            if not (target / member.name).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe source archive path')
            archive.extract(member, target, filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if sys.platform != 'linux' or sys.version_info < (3, 10):
        parser.error('Use Linux with Python >=3.10')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if not hasattr(tarfile, 'data_filter'):
        parser.error('Update Python to a patch release with tarfile extraction filters')
    for command in ['apt-get', 'apt-cache', 'dpkg-deb', 'pkg-config', 'cmake', 'c++']:
        if not shutil.which(command):
            parser.error(f'Missing prerequisite: {command}')
    release = Path('/etc/os-release').read_text()
    if 'VERSION_ID="22.04"' not in release:
        parser.error('This isolated dependency recipe is verified for Ubuntu 22.04')
    root = Path(args.prefix).expanduser().resolve()
    source, deps = root / 'source', root / 'deps'
    downloads = deps / 'packages'
    downloads.mkdir(parents=True, exist_ok=True)
    if not (source / 'REVISION').exists():
        extract_source(f'https://codeload.github.com/peteanderson80/Matterport3DSimulator/tar.gz/{REVISION}', source)
        (source / 'REVISION').write_text(REVISION)
    if (source / 'REVISION').read_text().strip() != REVISION:
        raise ValueError('Source prefix contains another MatterSim revision')
    if not (source / 'pybind11/CMakeLists.txt').exists():
        extract_source('https://codeload.github.com/pybind/pybind11/tar.gz/refs/tags/v2.13.6', source / 'pybind11')
    packages = {'libopencv-core-dev', 'libopencv-imgproc-dev', 'libopencv-imgcodecs-dev',
        'libopencv-core4.5d', 'libopencv-imgproc4.5d', 'libopencv-imgcodecs4.5d',
        'libjsoncpp-dev', 'libjsoncpp25', 'libglm-dev', 'libepoxy-dev', 'libepoxy0',
        'libtbb2', 'libgdcm3.0', 'libopenexr25', 'libgdal30', 'libopengl-dev',
        'libopengl0', 'libegl-dev', 'libegl1', 'libgl-dev', 'libgl1', 'libtiff5'}
    installed = subprocess.check_output(['dpkg-query', '-W', '-f=${binary:Package} ${Status}\n'], text=True)
    installed = {line.split()[0].split(':')[0] for line in installed.splitlines() if line.endswith('install ok installed')}
    pending, seen = list(packages), set()
    while pending:
        package = pending.pop()
        if package in seen:
            continue
        seen.add(package)
        if package in installed and package not in packages:
            continue
        metadata = subprocess.check_output(['apt-cache', 'depends', package], text=True)
        pending += [dependency for dependency in re.findall(r'^\s+Depends: ([a-zA-Z0-9.+-]+)\s*$', metadata, re.M)
                    if dependency not in installed and dependency not in seen]
    subprocess.run(['apt-get', 'download', *sorted(seen - installed | packages)], cwd=downloads, check=True)
    manifest = []
    for package in sorted(downloads.glob('*.deb')):
        subprocess.run(['dpkg-deb', '--extract', str(package), str(deps)], check=True)
        manifest.append({'file': package.name, 'sha256': hashlib.sha256(package.read_bytes()).hexdigest()})
    lib = deps / 'usr/lib/x86_64-linux-gnu'
    for pc in lib.glob('pkgconfig/*.pc'):
        pc.write_text(pc.read_text().replace('prefix=/usr', f'prefix={deps}/usr'))
    # OpenCV 4 removed a legacy name; include only modules actually used.
    includes = '#include <opencv2/core.hpp>\n#include <opencv2/imgproc.hpp>\n#include <opencv2/imgcodecs.hpp>'
    for filename in ['include/MatterSim.hpp', 'include/NavGraph.hpp', 'src/lib/NavGraph.cpp']:
        path = source / filename
        path.write_text(path.read_text().replace('#include <opencv2/opencv.hpp>', includes)
                        .replace('CV_LOAD_IMAGE_ANYDEPTH', 'cv::IMREAD_ANYDEPTH'))
    path = source / 'include/NavGraph.hpp'
    if '#include <list>' not in path.read_text():
        path.write_text('#include <list>\n#include <unordered_map>\n' + path.read_text())
    config = deps / 'opencv-config'
    config.mkdir(exist_ok=True)
    (config / 'OpenCVConfig.cmake').write_text(
        f'set(OpenCV_FOUND TRUE)\nset(OpenCV_INCLUDE_DIRS "{deps}/usr/include/opencv4")\n'
        'include_directories(${OpenCV_INCLUDE_DIRS})\nset(OpenCV_LIBS "' +
        ';'.join(str(lib / f'libopencv_{module}.so') for module in ['core', 'imgproc', 'imgcodecs']) + '")\n')
    env = dict(os.environ)
    runtime_libs = f'{lib}:{deps}/usr/lib'
    env.update(PATH=str(Path(sys.executable).parent) + ':' + env['PATH'],
        CMAKE_PREFIX_PATH=str(deps / 'usr'), PKG_CONFIG_PATH=str(lib / 'pkgconfig'),
        CPLUS_INCLUDE_PATH=str(deps / 'usr/include'), LIBRARY_PATH=runtime_libs,
        LD_LIBRARY_PATH=runtime_libs + ':' + env.get('LD_LIBRARY_PATH', ''),
        LD_PRELOAD=str(lib / 'libtiff.so.5'))
    build = source / 'build'
    subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-DEGL_RENDERING=ON',
        '-DCMAKE_BUILD_TYPE=Release', f'-DOpenCV_DIR={config}', f'-DPYTHON_EXECUTABLE={sys.executable}'], env=env, check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'MatterSimPython', f'-j{args.jobs}'], env=env, check=True)
    env['PYTHONPATH'] = str(build) + ':' + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-c', 'import MatterSim; import cv2; import torch; print("MatterSim import verified")'], env=env, check=True)
    (root / 'activate.sh').write_text(
        f'export PYTHONPATH={shlex.quote(str(build))}:"${{PYTHONPATH:-}}"\n'
        f'export LD_LIBRARY_PATH={shlex.quote(runtime_libs)}:"${{LD_LIBRARY_PATH:-}}"\n'
        f'export LD_PRELOAD={shlex.quote(str(lib / "libtiff.so.5"))}\n')
    (root / 'installation.json').write_text(json.dumps({'mattersim_revision': REVISION,
        'pybind11': '2.13.6', 'python': sys.version, 'packages': manifest,
        'source_changes': ['OpenCV 4 image-load constant', 'explicit C++/OpenCV includes']}, indent=2))
    print(f'Ready. Source {root / "activate.sh"} before running MatterSim.')


if __name__ == '__main__':
    main()
