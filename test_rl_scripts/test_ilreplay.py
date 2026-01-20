import configs
import os
import argparse
import json
from loguru import logger
import policy.utils as ml_utils
from data_utils.utils import set_seed, load_data, save_example_data
from data_utils.data_loader import get_dataloader
from configs.loader import ConfigLoader
from policy.policy_loader import (
    get_policy_data_processor,
    get_policy_data_collator,
    get_policy_trainer_class,
    load_policy_model_for_training,
)
from policy.trainer import BaseTrainer
from policy.rl.replay_buffer import (
    ILReplayBuffer,
    ReplayBufferDataLoader,
    transition_to_sample,
    sample_processed,
    verify_data_consistency,
)
import torch
import numpy as np
try:
    torch.serialization.add_safe_globals([np.ndarray])
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    torch.serialization.add_safe_globals([np.dtype])
    torch.serialization.safe_globals([np.dtype])
except:
    pass


def parse_param():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a policy model with RL replay buffer')
    parser.add_argument('-p', '--policy', type=str, default='act',
                       help='Policy config (name under configs/policy or absolute path to yaml)')
    parser.add_argument('-t', '--task', type=str, default='sim_transfer_cube_scripted',
                       help='Task config (name under configs/task or absolute path to yaml)')
    parser.add_argument('-c', '--training_config', type=str, default='default',
                       help='Training config (name under configs/training or absolute path to yaml)')
    parser.add_argument('-o', '--output_dir', type=str, default='ckpt/training_output',
                       help='Output directory for checkpoints')
    parser.add_argument('--eval_ratio', type=float, default=0.0,
                       help='Ratio of training data to use for evaluation')
    args, unknown = parser.parse_known_args()
    setattr(args, 'unknown_args', unknown)
    return args


def load_all_configs(args):
    """Load all configurations."""
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'unknown_args', []))
    task_config, task_cfg_path = cfg_loader.load_task(args.task)
    policy_config, policy_cfg_path = cfg_loader.load_policy(args.policy)
    training_config, training_args, training_cfg_path = cfg_loader.load_training(args.training_config, hyper_args=args)
    ConfigLoader.merge_all_parameters(task_config, policy_config, training_config, args)
    config_paths = {
        'task': task_cfg_path,
        'policy': policy_cfg_path,
        'training': training_cfg_path
    }
    return task_config, policy_config, training_args, config_paths


def create_replay_buffer_from_dataset(
    data_dict: dict,
    task_config: dict,
    policy_config: dict,
    capacity: int = None,
    use_train: bool = True,
    use_eval: bool = False,
    device: str = "cuda:0",
    storage_device: str = "cpu",
    batch_size: int = 1000,
    gc_interval: int = 2000,
    store_raw: bool = True,
) -> ILReplayBuffer:
    """
    Create and initialize a replay buffer from ILStudio dataset.
    
    NEW BEHAVIOR (store_raw=True, default):
    - Stores RAW data from the underlying dataset (bypasses normalizers/transforms)
    - Normalizers and transforms are extracted from the wrapped dataset
    - Processing (normalization, transforms) is applied on-demand during sampling
    - This provides flexibility for runtime data augmentation changes
    
    LEGACY BEHAVIOR (store_raw=False):
    - Data from load_data() is already normalized/transformed
    - Stored directly without changes
    
    Args:
        data_dict: Dict with 'train' and/or 'eval' datasets
        task_config: Task configuration
        policy_config: Policy configuration
        capacity: Buffer capacity (None = auto from dataset size)
        use_train: Include training data
        use_eval: Include evaluation data
        device: Device for sampling
        storage_device: Device for storage
        batch_size: Batch size for loading
        gc_interval: GC frequency during loading
        store_raw: If True, store raw data and apply processing on sampling
        
    Returns:
        ILReplayBuffer instance
    """
    train_data = data_dict.get('train') if use_train else None
    eval_data = data_dict.get('eval') if use_eval else None

    total_size = 0
    if train_data is not None:
        total_size += len(train_data)
    if eval_data is not None:
        total_size += len(eval_data)

    if total_size == 0:
        raise ValueError("No data to load into replay buffer")

    if capacity is None:
        capacity = total_size

    chunk_size = policy_config.get('chunk_size', 1)
    if chunk_size is None:
        chunk_size = policy_config.get('model_args', {}).get('chunk_size', 1)

    ctrl_space = policy_config.get('ctrl_space', 'ee')
    ctrl_type = policy_config.get('ctrl_type', 'delta')

    logger.info("="*60)
    logger.info("Creating Replay Buffer from Dataset")
    logger.info(f"  Capacity: {capacity}, Chunk size: {chunk_size}")
    logger.info(f"  Control: {ctrl_space}/{ctrl_type}, Total size: {total_size}")
    logger.info(f"  Store raw data: {store_raw}")
    logger.info("="*60)

    if train_data is not None:
        # from_ilstudio_dataset will automatically extract normalizers and transforms
        # from the wrapped dataset layers (NormalizedMapDataset, MapTransformPipeline)
        replay_buffer = ILReplayBuffer.from_ilstudio_dataset(
            raw_dataset=train_data,
            capacity=capacity,
            chunk_size=chunk_size,
            # Normalizers and transforms will be auto-extracted from wrapped dataset
            action_normalizer=None,
            state_normalizer=None,
            transforms=None,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            device=device,
            storage_device=storage_device,
            show_progress=True,
            batch_size=batch_size,
            gc_interval=gc_interval,
            store_raw=store_raw,
        )
    else:
        replay_buffer = ILReplayBuffer(
            capacity=capacity,
            chunk_size=chunk_size,
            action_normalizer=None,
            state_normalizer=None,
            transforms=None,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            device=device,
            storage_device=storage_device,
            store_raw=store_raw,
        )

    if eval_data is not None and use_eval:
        logger.info(f"Adding evaluation data ({len(eval_data)} samples)...")
        for i in range(len(eval_data)):
            sample = eval_data[i]
            obs = ILReplayBuffer._sample_to_metaobs(sample, ctrl_space)
            
            if i < len(eval_data) - 1:
                next_sample = eval_data[i + 1]
                if sample.get('episode_id', -1) != next_sample.get('episode_id', -1):
                    next_obs, done = obs, True
                else:
                    next_obs = ILReplayBuffer._sample_to_metaobs(next_sample, ctrl_space)
                    done = False
            else:
                next_obs, done = obs, True

            action_data = sample.get('action')
            if action_data is not None:
                if isinstance(action_data, torch.Tensor):
                    action_data = action_data.numpy()
                if action_data.ndim == 2 and chunk_size == 1:
                    action_data = action_data[0]
                from benchmark.base import MetaAction
                action = MetaAction(action=action_data.astype(np.float32), ctrl_space=ctrl_space, ctrl_type=ctrl_type)
                reward = sample.get('reward', 0.0)
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                # Note: For raw storage, data from environment is also stored raw
                replay_buffer.add(obs, action, reward, next_obs, done, truncated=False, already_normalized=not store_raw)

    logger.info(f"Replay Buffer created: {replay_buffer.size} transitions")
    logger.info(f"  Has action_normalizer: {replay_buffer.action_normalizer is not None}")
    logger.info(f"  Has state_normalizer: {replay_buffer.state_normalizer is not None}")
    logger.info(f"  Has transforms: {replay_buffer.transforms is not None}")
    return replay_buffer


def main(args):
    """Main training function with replay buffer."""
    args.is_training = True
    task_config, policy_config, training_args, config_paths = load_all_configs(args)

    seed = getattr(training_args, 'seed', 0)
    seed=1
    set_seed(seed)
    logger.info(f"🌱 Seed: {seed}")

    os.makedirs(training_args.output_dir, exist_ok=True)
    all_ckpts = [os.path.join(training_args.output_dir, ckpt_name) 
                 for ckpt_name in os.listdir(training_args.output_dir) 
                 if ckpt_name.startswith('checkpoint-') and os.path.isdir(os.path.join(training_args.output_dir, ckpt_name))]
    if len(all_ckpts) == 0:
        training_args.resume_from_checkpoint = None

    # Save policy metadata
    metadata_path = os.path.join(training_args.output_dir, 'policy_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'policy_module': policy_config['module_path'],
            'policy_name': policy_config['name'],
        }, f, indent=2)

    # Load model
    logger.info(f"Loading policy: {config_paths['policy']}")
    model_components = load_policy_model_for_training(config_paths['policy'], args, task_config)
    model = model_components['model']
    ml_utils.print_model_trainable_information(model)

    # Load dataset
    data_dict = load_data(args, task_config)
    train_data, val_data = data_dict['train'], data_dict['eval']

    # Get processor and collator
    data_processor = get_policy_data_processor(config_paths['policy'], args, model_components)
    data_collator = get_policy_data_collator(config_paths['policy'], args, model_components)

    # ===========================================================================
    # Create standard train_loader (train.py style) for comparison
    # ===========================================================================
    standard_train_loader, _ = get_dataloader(train_data, val_data, data_processor, data_collator, args)
    logger.info(f"Standard train_loader created with {len(standard_train_loader)} batches")

    # ===========================================================================
    # Create replay buffer
    # ===========================================================================
    replay_buffer = create_replay_buffer_from_dataset(
        data_dict=data_dict,
        task_config=task_config,
        policy_config=policy_config,
        capacity=None,
        use_train=True,
        use_eval=False,
        device='cuda:0',
        storage_device='cpu',
    )
    logger.info(f"Replay buffer: {replay_buffer.size} transitions")

    # ===========================================================================
    # Verify data consistency between train_data and replay_buffer
    # ===========================================================================
    # Get chunk_size from policy config for verification
    chunk_size = policy_config.get('chunk_size', 1)
    if chunk_size is None:
        chunk_size = policy_config.get('model_args', {}).get('chunk_size', 1)
    
    verify_results = verify_data_consistency(
        train_data=train_data,  # Pass original dataset directly
        replay_buffer=replay_buffer,
        data_processor=data_processor,
        sample_indices=[0, 1, 2, 10, 50],  # Check several indices
        tolerance=1e-5,
        chunk_size=chunk_size,  # Use policy's chunk_size for verification
    )
    
    if not verify_results['passed']:
        logger.warning("Data consistency check failed! Review mismatches above.")
    
    # ===========================================================================
    # Demo: Sample processed data directly
    # ===========================================================================
    logger.info("\n--- Demo: sample_processed() ---")
    processed_samples = sample_processed(replay_buffer, batch_size=4, data_processor=data_processor, device='cuda:0')
    logger.info(f"Got {len(processed_samples)} processed samples")
    for i, s in enumerate(processed_samples[:2]):
        logger.info(f"Sample {i} keys: {list(s.keys())}")
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: shape={v.shape}, device={v.device}")

    # ===========================================================================
    # Create ReplayBufferDataLoader
    # ===========================================================================
    # Calculate batches per epoch: same logic as standard DataLoader
    # (replay_buffer.size // batch_size), but the OOM issue was caused by
    # insufficient GC, which is now fixed in ReplayBufferDataLoader.
    # This ensures epoch counting matches train.py behavior.
    num_batches_per_epoch = max(1, replay_buffer.size // training_args.per_device_train_batch_size)
    logger.info(f"Using num_batches_per_epoch={num_batches_per_epoch} (based on dataset size)")
    
    train_loader = ReplayBufferDataLoader(
        replay_buffer=replay_buffer,
        batch_size=training_args.per_device_train_batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        data_processor=data_processor,
        data_collator=data_collator,
        device='cuda:0',
        gc_interval=10,
        apply_normalization=True,  # Apply normalization on sampling (for raw data)
        apply_transforms=True,  # Apply transforms on sampling (for data augmentation)
        chunk_size=chunk_size,  # Use policy's chunk_size for action chunk building
    )
    logger.info(f"ReplayBufferDataLoader: {len(train_loader)} batches/epoch, chunk_size={chunk_size}")
    logger.info(f"  Data pipeline info: {train_loader.get_data_info()}")

    # Test batch
    test_batch = next(iter(train_loader))
    logger.info(f"Test batch keys: {list(test_batch.keys())}")
    for k, v in test_batch.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Create eval loader (ReplayBufferDataLoader to match train format)
    eval_loader = None
    if val_data is not None:
        eval_buffer = create_replay_buffer_from_dataset(
            data_dict={'train': val_data, 'eval': None},
            task_config=task_config,
            policy_config=policy_config,
            capacity=None,
            use_train=True,
            use_eval=False,
            device='cuda:0',
            storage_device='cpu',
        )
        # For eval, use full dataset size (same as train)
        num_eval_batches = max(1, eval_buffer.size // training_args.per_device_train_batch_size)
        eval_loader = ReplayBufferDataLoader(
            replay_buffer=eval_buffer,
            batch_size=training_args.per_device_train_batch_size,
            num_batches_per_epoch=num_eval_batches,
            data_processor=data_processor,
            data_collator=data_collator,
            device='cuda:0',
            gc_interval=50,
            chunk_size=chunk_size,  # Use policy's chunk_size
        )
        logger.info(f"ReplayBuffer EvalLoader: {len(eval_loader)} batches/epoch")
    else:
        # Fallback: use training data for eval loss if no validation set
        logger.warning("No eval dataset found; using training data for eval loss.")
        # Use same num_batches as train_loader
        eval_loader = ReplayBufferDataLoader(
            replay_buffer=replay_buffer,
            batch_size=training_args.per_device_train_batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            data_processor=data_processor,
            data_collator=data_collator,
            device='cuda:0',
            gc_interval=10,
            chunk_size=chunk_size,  # Use policy's chunk_size
        )

    # Save example data
    save_example_data(train_data, training_args.output_dir)

    # ===========================================================================
    # Train
    # ===========================================================================
    train_class = get_policy_trainer_class(config_paths['policy']) or BaseTrainer
    trainer = train_class(
        args=training_args,
        model=model,
        tokenizer=model_components.get('tokenizer', None),
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    if trainer.is_world_process_zero():
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == '__main__':
    args = parse_param()
    main(args)
