"""Check the installed distribution without a source checkout on sys.path."""
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as workdir:
    subprocess.run([sys.executable, '-I', '-c', '''
import importlib.metadata
import sys
import vlastudio as vla
from vlastudio.paths import package_root

assert vla.__version__ == importlib.metadata.version('vlastudio')
assert 'torch' not in sys.modules
root = package_root()
assert 'site-packages' in root.parts, root
for handle in (
    vla.load_policy('act', check_dependencies=False),
    vla.load_dataset('sim_transfer_cube_scripted', check_dependencies=False),
    vla.load_env('aloha_transfer', check_dependencies=False),
):
    assert handle.config_path.is_file(), handle
    assert root in handle.config_path.parents, handle
assert (root / 'environments' / 'base.yaml').is_file()
assert list((root / 'benchmark' / 'aloha' / 'assets').rglob('*.xml'))
assert list((root / 'benchmark' / 'aloha' / 'assets').rglob('*.stl'))
assert vla.connect_policy('127.0.0.1:8000').address == '127.0.0.1:8000'
print('Installed package, version, aliases, assets and remote policy: PASS')
'''], cwd=workdir, check=True)
    subprocess.run([sys.executable, '-I', '-m', 'vlastudio', '--help'],
                   cwd=workdir, check=True)
    subprocess.run([sys.executable, '-I', '-m', 'vlastudio', 'env', 'create',
                    '--policy', 'openpi', '-n', 'pi05', '--dry-run'], cwd=workdir, check=True)
    subprocess.run([sys.executable, '-I', '-m', 'vlastudio', 'env', 'run', 'base',
                    '--cache-dir', workdir, '--', 'python', '-c',
                    'import vlastudio,sys; assert "torch" not in sys.modules'], cwd=workdir, check=True)
