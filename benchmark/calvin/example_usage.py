"""
Example usage of CALVIN environment in ILStudio

This script demonstrates how to:
1. Create a CALVIN environment
2. Run a simple policy
3. Evaluate on multiple sequences
"""

import sys
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

# Add parent directory to path
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.calvin import create_env, evaluate_calvin
from benchmark.base import MetaAction
from dataclasses import asdict


class RandomPolicy:
    """
    A simple random policy for demonstration.
    In practice, you would use your trained policy here.
    """
    def __init__(self, action_scale=0.01):
        self.action_scale = action_scale
        
    def select_action(self, obs, t):
        """
        Select an action based on observation.
        
        Args:
            obs: Dictionary with 'state', 'image', 'raw_lang' keys
            t: Current timestep
            
        Returns:
            Action dictionary
        """
        # Generate random action
        action = MetaAction(
            ctrl_space='ee',
            ctrl_type='delta',
            action=np.random.randn(7).astype(np.float32) * self.action_scale,
            gripper_continuous=False
        )
        
        # Random gripper state
        action.action[-1] = np.random.choice([0.0, 1.0])
        
        return asdict(action)
    
    def reset(self):
        """Reset policy state (if needed)."""
        pass


def example_single_sequence():
    """Example: Run a single sequence."""
    print("\n" + "="*60)
    print("Example 1: Single Sequence Evaluation")
    print("="*60)
    
    # Create configuration
    config = OmegaConf.create({
        'task': 'task_D',
        'show_gui': False,
        'num_sequences': 100,
        'sequence_idx': 0,
    })
    
    # Create environment
    env = create_env(config)
    policy = RandomPolicy(action_scale=0.01)
    
    print(f"\nTask: {env.task_name}")
    print(f"Sequence {env.sequence_idx}: {env.eval_sequence}")
    print(f"Initial instruction: '{env.get_current_language()}'")
    
    # Reset environment
    obs = env.reset()
    print(f"\nObservation shapes:")
    print(f"  State: {obs['state'].shape}")
    print(f"  Image: {obs['image'].shape}")
    
    # Run for limited steps
    max_steps_per_subtask = 50
    total_steps = 0
    done = False
    
    while not done and total_steps < max_steps_per_subtask * 5:
        # Get action from policy
        action = policy.select_action(obs, total_steps)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_steps += 1
        
        # Check if subtask completed
        if info['success']:
            print(f"\n✓ Subtask {info['current_subtask_idx']} completed!")
            print(f"  New instruction: '{env.get_current_language()}'")
        
        # Check if sequence done
        if done:
            print(f"\n✓ Sequence complete!")
            print(f"  Total subtasks completed: {info['subtasks_completed']}/5")
            break
    
    if not done:
        print(f"\n✗ Sequence incomplete after {total_steps} steps")
        print(f"  Subtasks completed: {env.subtasks_completed}/5")
    
    env.close()


def example_multiple_sequences():
    """Example: Evaluate on multiple sequences."""
    print("\n" + "="*60)
    print("Example 2: Multiple Sequence Evaluation")
    print("="*60)
    
    # Create policy
    policy = RandomPolicy(action_scale=0.01)
    
    # Create args
    args = OmegaConf.create({
        'task': 'task_D',
        'max_timesteps': 50,  # Reduced for demo
    })
    
    print(f"\nEvaluating on 5 sequences with random policy...")
    print("(Note: Random policy will likely complete 0 tasks)\n")
    
    # Run evaluation
    results = evaluate_calvin(
        args=args,
        policy=policy,
        env_class=create_env,
        num_sequences=5,
        max_steps_per_subtask=50,
    )
    
    print("\nResults:")
    print(f"  Per-sequence completions: {results['results']}")
    print(f"  Average sequence length: {results['avg_sequence_length']:.2f}")
    print(f"  Success rates:")
    for i, rate in enumerate(results['success_rates'], 1):
        print(f"    {i}/5: {rate:.1%}")


def example_with_your_policy():
    """
    Example: How to integrate your own policy.
    
    Replace RandomPolicy with your actual policy implementation.
    """
    print("\n" + "="*60)
    print("Example 3: Using Your Own Policy")
    print("="*60)
    
    print("""
To use your own policy:

1. Wrap your policy in a class with select_action() and reset() methods:

    class YourPolicyWrapper:
        def __init__(self, model, config):
            self.model = model
            self.config = config
            
        def select_action(self, obs, t):
            # obs is a dict with keys: 'state', 'image', 'raw_lang'
            # Extract what your model needs
            state = obs['state']  # (15,)
            image = obs['image']  # (2, 3, H, W)
            lang = obs['raw_lang']  # str
            
            # Run your model
            action = self.model(state, image, lang)
            
            # Return as MetaAction dict
            return {
                'ctrl_space': 'ee',
                'ctrl_type': 'delta',
                'action': action,  # (7,) numpy array
                'gripper_continuous': False
            }
        
        def reset(self):
            # Reset any internal state
            pass

2. Create the policy:

    policy = YourPolicyWrapper(your_model, your_config)

3. Run evaluation:

    from benchmark.calvin import evaluate_calvin
    
    args = OmegaConf.create({'task': 'task_D', 'max_timesteps': 360})
    
    results = evaluate_calvin(
        args=args,
        policy=policy,
        env_class=create_env,
        num_sequences=1000,
        max_steps_per_subtask=360,
    )
    
    print(f"Average sequence length: {results['avg_sequence_length']:.2f}")
    """)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CALVIN Environment Usage Examples")
    print("="*60)
    
    try:
        # Example 1: Single sequence
        example_single_sequence()
        
        # Example 2: Multiple sequences
        example_multiple_sequences()
        
        # Example 3: Integration guide
        example_with_your_policy()
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

