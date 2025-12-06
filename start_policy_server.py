#!/usr/bin/env python3
"""
Policy Server Startup Script

This script starts a policy server that listens for observation data 
and returns predicted actions over a network connection.
"""
import configs  
import os
import signal
import sys
from loguru import logger
from data_utils.utils import set_seed, load_normalizers
from benchmark.base import MetaPolicy
from deploy.remote import PolicyServer

def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Start a policy server for inference')
    
    # Server arguments
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address to bind (default: 0.0.0.0 for all interfaces)')
    parser.add_argument('-p', '--port', type=int, default=5000,
                       help='Port to listen on (default: 5000)')
    
    # Model arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    
    # Direct checkpoint loading
    parser.add_argument('-m', '--model_name_or_path', type=str, 
                       default='ckpt/act_sim_transfer_cube_scripted_zscore_example',
                       help='Path to the model checkpoint (directory or specific checkpoint)')
    parser.add_argument('--dataset_id', type=str, default='',
                       help='Dataset ID to use (if multiple datasets, defaults to first)')
    
    # Model parameters (will be loaded from checkpoint config if not provided)
    parser.add_argument('--chunk_size', type=int, default=-1,
                       help='Actual chunk size for policy that will truncate each raw chunk')
    
    # Parse arguments
    args, _ = parser.parse_known_args()
    return args


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("⏸ Received interrupt signal, shutting down...")
    sys.exit(0)


if __name__=='__main__':
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    set_seed(0)
    args = parse_param()
    args.is_training = False
    
    logger.info("="*60)
    logger.info("🚀 Policy Server Startup")
    logger.info("="*60)
    
    # Load normalizers and model
    logger.info("📦 Loading model and normalizers...")
    logger.info(f"   Model path: {args.model_name_or_path}")
    logger.info(f"   Dataset ID: {args.dataset_id if args.dataset_id else '(first dataset)'}")
    logger.info(f"   Device: {args.device}")
    
    # Load normalizers
    normalizers, ctrl_space, ctrl_type = load_normalizers(args)
    args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type
    
    # Load policy directly from checkpoint
    logger.info(f"   ✓ Loading model from checkpoint: {args.model_name_or_path}")
    from policy.direct_loader import load_model_from_checkpoint
    model_components = load_model_from_checkpoint(args.model_name_or_path, args)
    model = model_components['model']
    config = model_components.get('config', None)
    if config:
        logger.info(f"   ✓ Loaded config from checkpoint: {type(config).__name__}")
    
    # Create policy
    # Ensure model is in evaluation mode
    model.eval()
    
    policy = MetaPolicy(
        policy=model, 
        chunk_size=args.chunk_size, 
        action_normalizer=normalizers['action'], 
        state_normalizer=normalizers['state'], 
        ctrl_space=ctrl_space, 
        ctrl_type=ctrl_type
    )
    logger.info(f"   ✓ Policy created with chunk_size={args.chunk_size}")
    
    # Create and start server
    logger.info("🌐 Starting Policy Server...")
    server = PolicyServer(policy, host=args.host, port=args.port)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("⏸ Server interrupted by user")
    except Exception as e:
        logger.error(f"✗ Server error: {e}")
    finally:
        server.stop()
        logger.info("👋 Server shutdown complete")