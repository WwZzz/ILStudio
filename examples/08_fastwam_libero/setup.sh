#!/usr/bin/env bash
# Reuse a cached model environment, or add requirements to the current interpreter.
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$example_dir/../.." && pwd)"
python_bin="${PYTHON:-python3.10}"
export VLASTUDIO_CACHE="${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}"
venv_dir="${VENV_DIR:-$VLASTUDIO_CACHE/envs/example08}"
pip_args=()
setup_args=()
install_system=1
use_current=0
install_training=0
while (($#)); do
    case "$1" in
        -h|--help)
            cat <<'HELP'
Usage: bash examples/08_fastwam_libero/setup.sh [options]
  -i, --index-url URL  Override the Python package index.
  --root PATH         Reuse an existing LIBERO checkout.
  --assets PATH       Reuse an existing LIBERO assets directory.
  --training          Also install training and dataset conversion dependencies.
  --skip-system       Skip apt packages when native dependencies are installed.
  --current           Reuse the active Python 3.10 environment; do not create a venv.
Environment: PYTHON (Python 3.10), VLASTUDIO_CACHE, VENV_DIR.
HELP
            exit 0 ;;
        -i|--index-url|--root|--assets)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                -i|--index-url) pip_args+=(--index-url "$2"); export PIP_INDEX_URL="$2" ;;
                --root) setup_args+=(--root "$2") ;;
                --assets) setup_args+=(--assets "$2") ;;
            esac
            shift 2 ;;
        --training) install_training=1; shift ;;
        --skip-system) install_system=0; shift ;;
        --current) use_current=1; shift ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != Linux ]]; then
    echo "This example requires Linux and an NVIDIA GPU." >&2
    exit 1
fi
if ((use_current)); then
    python_bin="${PYTHON:-python}"
elif [[ -x "$venv_dir/bin/python" ]]; then
    python_bin="$venv_dir/bin/python"
fi
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Set PYTHON to a Python 3.10 interpreter"'
if ((install_system)); then
    if ! command -v apt-get >/dev/null; then
        echo "Install the native dependencies listed in setup.sh, then use --skip-system." >&2
        exit 1
    fi
    apt=(apt-get)
    if ((EUID != 0)); then apt=(sudo apt-get); fi
    "${apt[@]}" update
    "${apt[@]}" install -y build-essential python3.10-dev python3.10-venv git \
        libmagickwand-6.q16-6 libgl1 libegl1 libopengl0
fi
mkdir -p "$VLASTUDIO_CACHE" "$(dirname -- "$venv_dir")"
export VLASTUDIO_CACHE="$(cd -- "$VLASTUDIO_CACHE" && pwd)"
if ((use_current)); then
    python="$python_bin"
    activation_file="$VLASTUDIO_CACHE/example08/activate-current.sh"
    mkdir -p "$(dirname -- "$activation_file")"
else
    if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
        if [[ -e "$venv_dir" ]]; then
            echo "Refusing to overwrite a directory that is not a venv: $venv_dir" >&2
            exit 1
        fi
        "$python_bin" -m venv "$venv_dir"
    fi
    venv_dir="$(cd -- "$venv_dir" && pwd)"
    python="$venv_dir/bin/python"
    activation_file="$venv_dir/activate.sh"
fi
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for this example"'
# Keep an installed VLAStudio; pip adds only missing or incompatible requirements.
package_args=()
if ! "$python" -c 'from importlib.metadata import version; from packaging.version import Version; assert Version(version("vlastudio")) >= Version("0.2.2")' 2>/dev/null; then
    package_args=(-e "$repo_root")
fi
"$python" -m pip install "${pip_args[@]}" "${package_args[@]}" -r "$example_dir/requirements.txt"

if ((install_training)); then
    "$python" -m pip install "${pip_args[@]}" 'deepspeed==0.18.7' 'pyarrow>=14,<22' 'ninja' 'tensorboard'
fi
"$python" "$example_dir/setup.py" "${setup_args[@]}"
"$python" -m pip install --no-deps -e "$VLASTUDIO_CACHE/sources/FastWAM"

# One activation command selects Python and the simulator configuration.
{
    if ((!use_current)); then printf 'source %q\n' "$venv_dir/bin/activate"; fi
    printf 'export VLASTUDIO_CACHE=%q\n' "$VLASTUDIO_CACHE"
    printf 'source %q\n' "$VLASTUDIO_CACHE/fastwam/env.sh"
} > "$activation_file"
printf '\nSetup complete. Run from the repository root:\n'
printf 'source %q\n' "$activation_file"
printf 'python examples/08_fastwam_libero/evaluate.py\n'
