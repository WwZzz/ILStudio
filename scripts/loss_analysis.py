#!/usr/bin/env python3
"""
Loss Convergence Analysis Script

Analyze training logs to determine convergence and generate visualizations.

Usage:
    python scripts/loss_analysis.py --log_dir <log_directory> [--ckpt_dir <ckpt_directory>]
    
Example:
    python scripts/loss_analysis.py --log_dir ckpt/asi_gaussian_splat_9ch_6drot_base_224_0212/log
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard is not installed. Please install it with: pip install tensorboard")
    sys.exit(1)


class LossConvergenceAnalyzer:
    """Analyzer for loss convergence behavior."""
    
    def __init__(self, log_dir: str, ckpt_dir: Optional[str] = None, 
                 window_size: int = 1000, convergence_threshold: float = 0.01,
                 min_stable_steps: int = 5000):
        """
        Args:
            log_dir: TensorBoard log directory
            ckpt_dir: Checkpoint directory (used to save results)
            window_size: Window size for moving average
            convergence_threshold: Convergence threshold (relative change rate)
            min_stable_steps: Minimum stable steps (used to determine convergence)
        """
        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else self.log_dir.parent
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.min_stable_steps = min_stable_steps
        
        # Ensure output directory exists
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
    def load_tensorboard_data(self, tag: str = "train/loss") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from TensorBoard event files.
        
        Returns:
            (steps, values): Step indices and corresponding loss values
        """
        event_files = list(self.log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            raise ValueError(f"No event files found in {self.log_dir}")
        
        print(f"Found {len(event_files)} event file(s)")
        
        # Load all event files
        all_steps = []
        all_values = []
        
        for event_file in event_files:
            print(f"Loading {event_file.name}...")
            ea = EventAccumulator(str(event_file.parent))
            ea.Reload()
            
            # Get all available scalar tags
            scalar_tags = ea.Tags()['scalars']
            print(f"Available scalar tags: {scalar_tags}")
            
            if tag not in scalar_tags:
                print(f"Warning: Tag '{tag}' not found. Available tags: {scalar_tags}")
                # Try using the first available loss-related tag
                loss_tags = [t for t in scalar_tags if 'loss' in t.lower()]
                if loss_tags:
                    tag = loss_tags[0]
                    print(f"Using tag: {tag}")
                else:
                    raise ValueError(f"Tag '{tag}' not found and no loss tags available")
            
            scalar_events = ea.Scalars(tag)
            
            for event in scalar_events:
                all_steps.append(int(event.step))
                all_values.append(float(event.value))
        
        # Sort and deduplicate (if repeated steps exist)
        data = pd.DataFrame({'step': all_steps, 'value': all_values})
        data = data.sort_values('step')
        data = data.drop_duplicates(subset=['step'], keep='last')
        
        steps = data['step'].values
        values = data['value'].values
        
        print(f"Loaded {len(steps)} data points")
        print(f"Step range: [{steps.min()}, {steps.max()}]")
        print(f"Value range: [{values.min():.6f}, {values.max():.6f}]")
        
        return steps, values
    
    def compute_moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Compute moving average."""
        if len(values) < window:
            return values
        return pd.Series(values).rolling(window=window, center=True).mean().values
    
    def detect_convergence(self, steps: np.ndarray, values: np.ndarray) -> Dict:
        """
        Detect convergence point.
        
        Returns:
            Dictionary containing convergence details
        """
        # Compute moving average
        ma_values = self.compute_moving_average(values, self.window_size)
        
        # Compute relative change (using moving average)
        # For each point, evaluate relative change in a following window
        convergence_info = {
            'converged': False,
            'convergence_step': None,
            'convergence_loss': None,
            'reason': '',
            'analysis': {}
        }
        
        # Method 1: detect stable interval (variation below threshold)
        # Search backward and take the first matching stable interval
        stable_start = None
        for i in range(len(ma_values) - self.min_stable_steps, 0, -100):  # check every 100 steps
            if i < 0:
                break
            end_idx = min(i + self.min_stable_steps, len(ma_values))
            window_ma = ma_values[i:end_idx]
            
            if len(window_ma) < 100:
                continue
                
            # Compute relative variation in the window
            window_mean = np.mean(window_ma)
            window_std = np.std(window_ma)
            relative_std = window_std / (window_mean + 1e-8)
            
            # If relative std is below threshold, consider it stable
            if relative_std < self.convergence_threshold:
                stable_start = i
                convergence_info['converged'] = True
                convergence_info['convergence_step'] = int(steps[i])
                convergence_info['convergence_loss'] = float(window_mean)
                convergence_info['reason'] = (
                    f"Found a stable interval near step {steps[i]} "
                    f"(relative std: {relative_std:.4f} < {self.convergence_threshold})"
                )
                break
        
        # Method 2: if method 1 fails, use loss drop rate
        if not convergence_info['converged']:
            # Compute loss drop rate
            # Split data into segments and compute average drop per segment
            num_segments = 10
            segment_size = len(values) // num_segments
            
            segment_means = []
            segment_steps = []
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(values)
                segment_means.append(np.mean(values[start_idx:end_idx]))
                segment_steps.append(steps[start_idx + (end_idx - start_idx) // 2])
            
            # Compute drop rate between adjacent segments
            drop_rates = []
            for i in range(len(segment_means) - 1):
                drop_rate = (segment_means[i] - segment_means[i+1]) / (segment_means[i] + 1e-8)
                drop_rates.append(drop_rate)
            
            # If recent segment drop rates are all small, likely converged
            if len(drop_rates) >= 3:
                recent_drop_rates = drop_rates[-3:]
                avg_recent_drop = np.mean(recent_drop_rates)
                
                if avg_recent_drop < 0.05:  # average drop rate < 5%
                    convergence_info['converged'] = True
                    convergence_info['convergence_step'] = int(segment_steps[-3])
                    convergence_info['convergence_loss'] = float(segment_means[-1])
                    convergence_info['reason'] = (
                        f"Loss drop rate decreases significantly near step {segment_steps[-3]} "
                        f"(average drop rate: {avg_recent_drop:.4f} < 0.05)"
                    )
        
        # Method 3: if loss is already low, use a looser convergence rule
        if not convergence_info['converged']:
            # Check the last 20% of the data
            last_20_percent_start = int(len(values) * 0.8)
            last_20_values = values[last_20_percent_start:]
            last_20_steps = steps[last_20_percent_start:]
            last_20_ma = ma_values[last_20_percent_start:]
            
            if len(last_20_values) > 100:
                # Compute mean loss and std over last 20%
                mean_loss = np.mean(last_20_ma)
                std_loss = np.std(last_20_ma)
                cv = std_loss / (mean_loss + 1e-8)  # coefficient of variation
                
                # If loss is low (<0.01) and variation is modest (<0.5), treat as converged
                if mean_loss < 0.01 and cv < 0.5:
                    convergence_info['converged'] = True
                    convergence_info['convergence_step'] = int(last_20_steps[0])
                    convergence_info['convergence_loss'] = float(mean_loss)
                    convergence_info['reason'] = (
                        f"Loss reaches a low level near step {last_20_steps[0]} "
                        f"(mean loss: {mean_loss:.6f} < 0.01, CV: {cv:.4f} < 0.5)"
                    )
        
        # Method 4: detect plateau phase (small changes across consecutive segments)
        if not convergence_info['converged']:
            num_segments = 20
            segment_size = len(values) // num_segments
            segment_means = []
            segment_steps = []
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(values)
                segment_means.append(np.mean(values[start_idx:end_idx]))
                segment_steps.append(steps[start_idx + (end_idx - start_idx) // 2])
            
            # Search backward for a continuous plateau
            for i in range(len(segment_means) - 2, max(0, len(segment_means) - 6), -1):
                # Check change across 3 consecutive segments
                recent_segments = segment_means[i:i+3]
                if len(recent_segments) == 3:
                    max_change = np.max(recent_segments) - np.min(recent_segments)
                    mean_value = np.mean(recent_segments)
                    relative_change = max_change / (mean_value + 1e-8)
                    
                    # If relative change < 10%, consider it a plateau
                    if relative_change < 0.1:
                        convergence_info['converged'] = True
                        convergence_info['convergence_step'] = int(segment_steps[i])
                        convergence_info['convergence_loss'] = float(mean_value)
                        convergence_info['reason'] = (
                            f"Entered a plateau near step {segment_steps[i]} "
                            f"(relative change: {relative_change:.4f} < 0.1)"
                        )
                        break
        
        # If still not found, mark as not converged
        if not convergence_info['converged']:
            convergence_info['reason'] = (
                "No sustained stable interval found "
                "(loss may still be decreasing or fluctuating heavily)"
            )
        
        # Compute key statistics
        initial_10p = values[:len(values)//10]
        final_10p = values[-len(values)//10:]
        
        convergence_info['analysis'] = {
            'min_loss': float(np.min(values)),
            'min_loss_step': int(steps[np.argmin(values)]),
            'final_mean_loss': float(np.mean(final_10p)),
            'final_std_loss': float(np.std(final_10p)),
            'initial_mean_loss': float(np.mean(initial_10p)),
            'overall_drop_ratio': float(1 - np.mean(final_10p) / (np.mean(initial_10p) + 1e-8)),
            'total_steps': int(steps.max())
        }
        
        return convergence_info
    
    def plot_analysis(self, steps: np.ndarray, values: np.ndarray, 
                     convergence_info: Dict, tag: str = "train/loss"):
        """Generate visualization plots."""
        # Create a simple loss curve plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Compute moving average
        ma_values = self.compute_moving_average(values, self.window_size)
        
        # Plot raw loss and moving average
        ax.plot(steps, values, alpha=0.3, color='blue', label='Raw Loss', linewidth=0.5)
        ax.plot(steps, ma_values, color='red', label=f'Moving Average (window={self.window_size})', linewidth=2)
        
        # Mark convergence point
        if convergence_info['convergence_step']:
            ax.axvline(x=convergence_info['convergence_step'], 
                      color='green', linestyle='--', linewidth=2, 
                      label=f"Convergence Step: {convergence_info['convergence_step']}")
            ax.axhline(y=convergence_info['convergence_loss'], 
                      color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Loss Curve: {tag}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Use logarithmic scale
        
        # Save plot image
        output_path = self.ckpt_dir / 'loss_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        plt.close()
        
        # Save raw data as CSV
        data_df = pd.DataFrame({
            'step': steps,
            'loss': values
        })
        csv_path = self.ckpt_dir / 'loss_data.csv'
        data_df.to_csv(csv_path, index=False)
        print(f"Saved data to: {csv_path}")
    
    def generate_report(self, convergence_info: Dict, tag: str = "train/loss") -> str:
        """Generate text report."""
        report = []
        report.append("=" * 60)
        report.append("Loss Convergence Analysis Report")
        report.append("=" * 60)
        report.append("")
        report.append(f"Tag: {tag}")
        report.append(f"Converged: {'YES' if convergence_info['converged'] else 'NO'}")
        report.append(f"Reason: {convergence_info['reason']}")
        report.append("")
        
        analysis = convergence_info['analysis']
        
        if convergence_info['converged']:
            report.append(f"Convergence step: {convergence_info['convergence_step']}")
            report.append(f"Convergence loss: {convergence_info['convergence_loss']:.6f}")
        else:
            report.append("Convergence step: Not detected")
            report.append("Convergence loss: N/A")
        
        report.append("")
        report.append("Key Statistics:")
        report.append(f"  Total steps: {analysis['total_steps']}")
        report.append(f"  Minimum loss: {analysis['min_loss']:.6f} @ step {analysis['min_loss_step']}")
        report.append(f"  Initial mean loss (first 10%): {analysis['initial_mean_loss']:.6f}")
        report.append(f"  Final mean loss (last 10%): {analysis['final_mean_loss']:.6f} ± {analysis['final_std_loss']:.6f}")
        report.append(f"  Overall drop ratio: {analysis['overall_drop_ratio']:.2%}")
        report.append("")
        
        return "\n".join(report)
    
    def run_analysis(self, tag: str = "train/loss"):
        """Run the full analysis pipeline."""
        print(f"Starting loss convergence analysis...")
        print(f"Log directory: {self.log_dir}")
        print(f"Output directory: {self.ckpt_dir}")
        print(f"Tag: {tag}")
        print("")
        
        # Load data
        steps, values = self.load_tensorboard_data(tag)
        
        # Detect convergence
        print("\nDetecting convergence...")
        convergence_info = self.detect_convergence(steps, values)
        
        # Generate visualizations
        print("\nGenerating plots...")
        self.plot_analysis(steps, values, convergence_info, tag)
        
        # Generate report
        print("\nGenerating report...")
        report = self.generate_report(convergence_info, tag)
        
        # Save report
        report_path = self.ckpt_dir / 'loss_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Saved report to: {report_path}")
        
        # Print report
        print("\n" + report)
        
        return convergence_info


def main():
    parser = argparse.ArgumentParser(description='Analyze loss convergence from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing TensorBoard event files')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                       help='Checkpoint directory to save results (default: parent of log_dir)')
    parser.add_argument('--tag', type=str, default='train/loss',
                       help='TensorBoard scalar tag to analyze (default: train/loss)')
    parser.add_argument('--window_size', type=int, default=1000,
                       help='Window size for moving average (default: 1000)')
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                       help='Convergence threshold for relative std (default: 0.01)')
    parser.add_argument('--min_stable_steps', type=int, default=5000,
                       help='Minimum stable steps for convergence (default: 5000)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = LossConvergenceAnalyzer(
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        window_size=args.window_size,
        convergence_threshold=args.convergence_threshold,
        min_stable_steps=args.min_stable_steps
    )
    
    # Run analysis
    analyzer.run_analysis(tag=args.tag)


if __name__ == '__main__':
    main()
