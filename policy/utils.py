from loguru import logger
from deploy.remote import PolicyClient, parse_server_address, is_server_address

def load_policy(args):
    # Check if model_name_or_path is a server address or local checkpoint
    if is_server_address(args.model_name_or_path):
        logger.info("="*60)
        logger.info("🤖 Remote Policy Evaluation")
        logger.info("="*60)
        # Remote server mode
        host, port = parse_server_address(args.model_name_or_path)
        logger.info(f"🌐 Using remote policy server: {host}:{port}")
        
        # Create remote policy client (no need for normalizers or local model)
        policy = PolicyClient(
            host=host,
            port=port,
            chunk_size=getattr(args, 'chunk_size', None),
        )
        
        # Set dummy values for compatibility
        # For real robot eval, these will be updated after policy is created
        if not hasattr(args, 'ctrl_space'):
            args.ctrl_space = policy.ctrl_space
            args.ctrl_type = policy.ctrl_type
        
    else:
        # Local model mode (fallback to original behavior)
        print("="*60)
        print("🤖 Local Policy Evaluation")
        print("="*60)
        # Load normalizers and model as before
        from data_utils.utils import load_normalizers
        from benchmark.base import MetaPolicy
        
        normalizers, ctrl_space, ctrl_type = load_normalizers(args)
        args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type
        
        # Load policy directly from checkpoint
        print(f"Loading model from checkpoint: {args.model_name_or_path}")
        # Fallback to direct checkpoint loading
        if not hasattr(args, 'policy_module'):
            from policy.direct_loader import load_model_from_checkpoint
            if not hasattr(args, 'is_training'):
                args.is_training = False
            model_components = load_model_from_checkpoint(args.model_name_or_path, args)
            model = model_components['model']
        else:
            from policy.direct_loader import load_model_from_checkpoint
            model_components = load_model_from_checkpoint(args.model_name_or_path, args)
            model = model_components['model']
            config = model_components.get('config', None)
            if config:
                print(f"Loaded config from checkpoint: {type(config).__name__}")
            policy = MetaPolicy(
                policy=model, 
                chunk_size=getattr(args, 'chunk_size', None), 
                action_normalizer=normalizers['action'], 
                state_normalizer=normalizers['state'], 
                ctrl_space=ctrl_space, 
                ctrl_type=ctrl_type,
                img_size = getattr(args, 'image_size', None)
            )
    return policy

def print_model_trainable_information(model, rank0_print=None):
    if rank0_print is None: rank0_print = logger.info
    lora_para = sum(p.numel() for n, p in model.named_parameters() if (p.requires_grad and 'lora' in n))
    all_para = sum(p.numel() for n, p in model.named_parameters())
    train_para = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    rank0_print(f"Lora parameters/trainalbe parameters/all parameters:{lora_para/1000000}M/{train_para/1000000}M/{(all_para-lora_para)/1000000}M")