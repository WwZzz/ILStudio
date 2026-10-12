"""Prepare the official Wan-derived action backbone before training."""
import os
from pathlib import Path
import subprocess
import sys
from omegaconf import OmegaConf
from vlastudio.policy.fastwam.training import model_config


def main():
    cache = Path(os.environ.get("VLASTUDIO_CACHE", "~/.cache/vlastudio")).expanduser().resolve()
    os.environ["VLASTUDIO_CACHE"] = str(cache)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(cache / "models/fastwam"))
    upstream = cache / "sources/FastWAM"
    output = cache / "models/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    if output.exists():
        print(f"Action backbone already exists: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = model_config(upstream, action_dit_pretrained_path=output)
    config = output.with_suffix(".yaml")
    OmegaConf.save(cfg, config, resolve=True)
    temporary = output.with_suffix(f".tmp.{os.getpid()}")
    subprocess.run([sys.executable, str(upstream / "scripts/preprocess_action_dit_backbone.py"),
                    "--model-config", str(config), "--output", str(temporary),
                    "--device", "cpu", "--dtype", "float32", "--apply-alpha-scaling", "true"],
                   cwd=upstream, check=True)
    temporary.replace(output)
    print(f"Prepared official action backbone: {output}")


if __name__ == "__main__":
    main()
