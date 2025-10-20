
from deploy.remote import PolicyClient, parse_server_address, is_server_address, WebSocketPolicyClient

def is_websocket_address(path):
    if path.startswith("https://"):
        path = "wss://" + path[8:]
    return path.startswith("ws://") or path.startswith("wss://"), path

def load_policy(args):
    model_path = args.model_name_or_path
    is_ws, ws_uri = is_websocket_address(model_path)

    # Check if model_name_or_path is a server address or local checkpoint
    if is_ws:
        print("="*60)
        print("🤖 Remote WebSocket Policy Evaluation")
        print("="*60)
        print(f"🌐 Using remote WebSocket policy server: {ws_uri}")
        
        policy = WebSocketPolicyClient(
            uri=ws_uri,
            chunk_size=args.chunk_size,
        )
        
        if not hasattr(args, 'ctrl_space'):
            args.ctrl_space = policy.ctrl_space
            args.ctrl_type = policy.ctrl_type
            
    elif is_server_address(args.model_name_or_path):
        print("="*60)
        print("🤖 Remote TCP Policy Evaluation")
        print("="*60)
        # Remote server mode
        host, port = parse_server_address(args.model_name_or_path)
        print(f"🌐 Using remote policy server: {host}:{port}")
        
        # Create remote policy client (no need for normalizers or local model)
        policy = PolicyClient(
            host=host,
            port=port,
            chunk_size=args.chunk_size,
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
        from policy.direct_loader import load_model_from_checkpoint
        model_components = load_model_from_checkpoint(args.model_name_or_path, args)
        model = model_components['model']
        config = model_components.get('config', None)
        if config:
            print(f"Loaded config from checkpoint: {type(config).__name__}")
        policy = MetaPolicy(
            policy=model, 
            chunk_size=args.chunk_size, 
            action_normalizer=normalizers['action'], 
            state_normalizer=normalizers['state'], 
            ctrl_space=ctrl_space, 
            ctrl_type=ctrl_type
        )
    return policy

def print_model_trainable_information(model, rank0_print=None):
    if rank0_print is None: rank0_print = print
    lora_para = sum(p.numel() for n, p in model.named_parameters() if (p.requires_grad and 'lora' in n))
    all_para = sum(p.numel() for n, p in model.named_parameters())
    train_para = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    rank0_print(f"Lora parameters/trainalbe parameters/all parameters:{lora_para/1000000}M/{train_para/1000000}M/{(all_para-lora_para)/1000000}M")