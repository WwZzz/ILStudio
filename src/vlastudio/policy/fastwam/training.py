"""FastWAM loss and checkpoint hooks for VLAStudio's BaseTrainer."""
import json
import os
from pathlib import Path
import shutil

import torch
from vlastudio.policy.trainer import BaseTrainer


def expanded(value):
    os.environ.setdefault("VLASTUDIO_CACHE", str(Path.home() / ".cache/vlastudio"))
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def model_config(upstream_root, *, random_init=False, load_text_encoder=False, action_dit_pretrained_path=None):
    from hydra import compose, initialize_config_dir
    from fastwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(expanded(upstream_root) / "configs")):
        cfg = compose(config_name="train", overrides=["task=libero_uncond_2cam224_1e-4"])
    cfg.model.redirect_common_files = False
    cfg.model.skip_dit_load_from_pretrain = random_init
    if random_init:
        cfg.model.action_dit_pretrained_path = None
    else:
        path = action_dit_pretrained_path or str(expanded(upstream_root) / cfg.model.action_dit_pretrained_path)
        cfg.model.action_dit_pretrained_path = str(expanded(path))
    cfg.model.load_text_encoder = load_text_encoder
    return cfg.model


class FastWAMTrainingPolicy(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        self.spec = dict(upstream_root=str(expanded(args.upstream_root)),
                         dataset_stats=str(expanded(args.dataset_stats)),
                         seed=int(getattr(args, "seed", 42)), num_inference_steps=10,
                         sigma_shift=float(getattr(args, "sigma_shift", 5.0)), compile_action_infer=False)
        self.initialization = str(getattr(args, "initialization", "official"))
        if self.initialization not in ("official", "random"):
            raise ValueError("initialization must be official or random")
        cfg = model_config(self.spec["upstream_root"], random_init=self.initialization == "random",
                           action_dit_pretrained_path=getattr(args, "action_dit_pretrained_path", None))
        if self.initialization == "official" and not Path(cfg.action_dit_pretrained_path).is_file():
            raise FileNotFoundError("Official ActionDiT initialization is missing: " + cfg.action_dit_pretrained_path)
        self.resolved_config = OmegaConf.to_container(cfg, resolve=True)
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        torch.cuda.set_device(device)
        self.core = instantiate(cfg, model_dtype=torch.bfloat16, device=device)
        self.core.requires_grad_(False)
        self.core.mot.requires_grad_(True)
        self.core.proprio_encoder.requires_grad_(True)
        self.ctrl_space, self.ctrl_type = "ee", "delta"
        self.chunk_size, self.action_dim = 32, 7
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        self.core.mot.train(mode)
        self.core.proprio_encoder.train(mode)
        return self

    def forward(self, video=None, action=None, proprio=None, image_is_pad=None,
                action_is_pad=None, context=None, context_mask=None, prompt=None, **kwargs):
        loss, metrics = self.core.training_loss(dict(video=video, action=action, proprio=proprio,
            image_is_pad=image_is_pad, action_is_pad=action_is_pad,
            context=context, context_mask=context_mask, prompt=prompt))
        return {"loss": loss, **metrics}

    def save_bundle(self, output_dir, step):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        weights = output / "fastwam.pt"
        self.core.save_checkpoint(str(weights), step=step)
        stats = output / "dataset_stats.json"
        if Path(self.spec["dataset_stats"]).resolve() != stats.resolve():
            shutil.copyfile(self.spec["dataset_stats"], stats)
        spec = dict(self.spec, weights="fastwam.pt", dataset_stats="dataset_stats.json",
                    model_config=self.resolved_config, initialization=self.initialization, training_step=step)
        (output / "fastwam.json").write_text(json.dumps(spec, indent=2))
        (output / "policy_metadata.json").write_text(json.dumps({"policy_module": "vlastudio.policy.fastwam", "policy_name": "fastwam"}))
        (output / "config.json").write_text(json.dumps({"ctrl_space": "ee", "ctrl_type": "delta", "chunk_size": 32, "action_dim": 7, "state_dim": 8}))
        (output / "normalize.json").write_text(json.dumps({"datasets": [{"dataset_id": "fastwam", "ctrl_space": "ee", "ctrl_type": "delta"}], "state": {"fastwam": "identity"}, "action": {"fastwam": "identity"}}))


class FastWAMTrainer(BaseTrainer):
    """Keep the shared training loop with the FastWAM optimizer, schedule and bundle format."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=self.args.learning_rate, betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            optimizer = optimizer if optimizer is not None else self.optimizer
            warmup = min(self.args.get_warmup_steps(num_training_steps), num_training_steps - 1)
            cosine = CosineAnnealingLR(optimizer, T_max=max(num_training_steps - warmup, 1),
                                       eta_min=self.args.learning_rate * 0.01)
            self.lr_scheduler = (SequentialLR(optimizer, [
                LinearLR(optimizer, start_factor=1.0 / warmup, total_iters=warmup), cosine],
                milestones=[warmup]) if warmup else cosine)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _save(self, output_dir=None, state_dict=None):
        self.model.save_bundle(output_dir or self.args.output_dir, self.state.global_step)
        torch.save(self.args, Path(output_dir or self.args.output_dir) / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = model if model is not None else self.model
        payload = torch.load(Path(resume_from_checkpoint) / "fastwam.pt", map_location="cpu", weights_only=True, mmap=True)
        target.core.mot.load_state_dict(payload["mot"], strict=True)
        target.core.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
