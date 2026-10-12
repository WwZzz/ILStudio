"""Map standard temporal VLAStudio samples to FastWAM training tensors."""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import default_collate
from torchvision.transforms import Resize


class FastWAMProcessor:
    def __init__(self, stats_path, text_cache):
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json, SingleFieldLinearNormalizer
        stats = load_dataset_stats_from_json(os.path.expandvars(os.path.expanduser(stats_path)))
        self.normalizers = {
            key: SingleFieldLinearNormalizer({name.removeprefix("global_"): value
                 for name, value in stats[key]["default"].items()}, mode="min/max")
            for key in ("action", "state")
        }
        self.text_cache = Path(os.path.expandvars(os.path.expanduser(text_cache)))
        self.resize = Resize((224, 224), antialias=True)

    @lru_cache(maxsize=256)
    def _context(self, prompt):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.text_cache / f"{digest}.t5_len128.wan22ti2v5b.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context, mask = payload["context"].clone(), payload["mask"].bool()
        if context.shape != (128, 4096) or mask.shape != (128,):
            raise ValueError(f"Unexpected cached text shape in {path}")
        context[~mask] = 0
        # Wan uses zeroed padding embeddings with an all-true attention mask.
        return context, torch.ones_like(mask)

    def __call__(self, sample):
        images = sample["image"]
        if images.ndim != 5 or images.shape[:3] != (2, 9, 3) or images.dtype != torch.uint8:
            raise ValueError(f"FastWAM training expects uint8 image[2,9,3,H,W], got {images.shape}/{images.dtype}")
        if sample["action"].shape != (32, 7) or sample["state"].shape != (32, 8):
            raise ValueError("FastWAM requires action[32,7] and state[32,8]")
        if sample["image_is_pad"].shape != (9,) or sample["is_pad"].shape != (32,):
            raise ValueError("Temporal padding masks must match image and action horizons")
        cameras = self.resize(images.reshape(-1, *images.shape[2:]).float() / 255.0)
        cameras = cameras.reshape(2, 9, 3, 224, 224)
        video = torch.cat(tuple(cameras), dim=-1).permute(1, 0, 2, 3)
        prompt = "A video recorded from a robot's point of view executing the following instruction: " + sample["raw_lang"]
        context, context_mask = self._context(prompt)
        return dict(video=(video - 0.5) / 0.5,
                    action=self.normalizers["action"].forward(sample["action"]),
                    proprio=self.normalizers["state"].forward(sample["state"]),
                    image_is_pad=sample["image_is_pad"], action_is_pad=sample["is_pad"],
                    context=context, context_mask=context_mask, prompt=prompt)


def collate(samples):
    return default_collate(samples)


def prepare_text_cache(dataset_root, cache_dir, device="cuda"):
    """Encode instructions from converted HDF5 episodes using the frozen Wan T5."""
    import json
    import h5py
    root, cache = Path(dataset_root), Path(cache_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    prompts = set()
    for episode in manifest["episodes"]:
        with h5py.File(root / episode["path"], "r") as f:
            prompts.add("A video recorded from a robot's point of view executing the following instruction: "
                        + f["language_instruction"].asstr()[0])
    cache.mkdir(parents=True, exist_ok=True)
    missing = {prompt: cache / (hashlib.sha256(prompt.encode("utf-8")).hexdigest()
               + ".t5_len128.wan22ti2v5b.pt") for prompt in sorted(prompts)}
    missing = {prompt: path for prompt, path in missing.items() if not path.exists()}
    if not missing:
        print(f"Text cache ready: {len(prompts)} instructions", flush=True)
        return
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    _, text, _, tokenizer_config = _resolve_configs("Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.1-T2V-1.3B", False)
    text.download_if_necessary()
    tokenizer_config.download_if_necessary()
    encoder = _load_registered_model(text.path, "wan_video_text_encoder",
                                     torch_dtype=torch.bfloat16, device=device).eval()
    encoder.requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=128, clean="whitespace")
    with torch.inference_mode():
        for prompt, path in missing.items():
            ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
            mask = mask.to(device=device, dtype=torch.bool)
            context = encoder(ids.to(device), mask)[0].to(device="cpu", dtype=torch.bfloat16)
            temporary = path.with_suffix(f".tmp.{os.getpid()}")
            torch.save({"context": context.contiguous(), "mask": mask[0].cpu().contiguous()}, temporary)
            temporary.replace(path)
    print(f"Cached {len(missing)} new instructions", flush=True)
