#!/usr/bin/env python3
"""
Loss Convergence Analysis Script

分析训练日志，判断模型是否收敛，并生成可视化图表。

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
    """分析loss收敛情况的类"""
    
    def __init__(self, log_dir: str, ckpt_dir: Optional[str] = None, 
                 window_size: int = 1000, convergence_threshold: float = 0.01,
                 min_stable_steps: int = 5000):
        """
        Args:
            log_dir: TensorBoard日志目录
            ckpt_dir: 检查点目录（用于保存结果）
            window_size: 用于计算移动平均的窗口大小
            convergence_threshold: 收敛阈值（相对变化率）
            min_stable_steps: 最小稳定步数（用于判断收敛）
        """
        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else self.log_dir.parent
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.min_stable_steps = min_stable_steps
        
        # 确保输出目录存在
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
    def load_tensorboard_data(self, tag: str = "train/loss") -> Tuple[np.ndarray, np.ndarray]:
        """
        从TensorBoard事件文件加载数据
        
        Returns:
            (steps, values): 步数和对应的loss值
        """
        event_files = list(self.log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            raise ValueError(f"No event files found in {self.log_dir}")
        
        print(f"Found {len(event_files)} event file(s)")
        
        # 加载所有事件文件
        all_steps = []
        all_values = []
        
        for event_file in event_files:
            print(f"Loading {event_file.name}...")
            ea = EventAccumulator(str(event_file.parent))
            ea.Reload()
            
            # 获取所有可用的scalar tags
            scalar_tags = ea.Tags()['scalars']
            print(f"Available scalar tags: {scalar_tags}")
            
            if tag not in scalar_tags:
                print(f"Warning: Tag '{tag}' not found. Available tags: {scalar_tags}")
                # 尝试使用第一个可用的loss相关tag
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
        
        # 排序并去重（如果有重复的step）
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
        """计算移动平均"""
        if len(values) < window:
            return values
        return pd.Series(values).rolling(window=window, center=True).mean().values
    
    def detect_convergence(self, steps: np.ndarray, values: np.ndarray) -> Dict:
        """
        检测收敛点
        
        Returns:
            包含收敛信息的字典
        """
        # 计算移动平均
        ma_values = self.compute_moving_average(values, self.window_size)
        
        # 计算相对变化率（使用移动平均）
        # 对于每个点，计算后续一段窗口内的相对变化
        convergence_info = {
            'converged': False,
            'convergence_step': None,
            'convergence_loss': None,
            'reason': '',
            'analysis': {}
        }
        
        # 方法1: 检测稳定区间（变化率小于阈值）
        # 从后往前找，找到第一个满足条件的稳定区间
        stable_start = None
        for i in range(len(ma_values) - self.min_stable_steps, 0, -100):  # 每100步检查一次
            if i < 0:
                break
            end_idx = min(i + self.min_stable_steps, len(ma_values))
            window_ma = ma_values[i:end_idx]
            
            if len(window_ma) < 100:
                continue
                
            # 计算窗口内的相对变化率
            window_mean = np.mean(window_ma)
            window_std = np.std(window_ma)
            relative_std = window_std / (window_mean + 1e-8)
            
            # 如果相对标准差小于阈值，认为稳定
            if relative_std < self.convergence_threshold:
                stable_start = i
                convergence_info['converged'] = True
                convergence_info['convergence_step'] = int(steps[i])
                convergence_info['convergence_loss'] = float(window_mean)
                convergence_info['reason'] = f"在step {steps[i]}附近发现稳定区间（相对标准差: {relative_std:.4f} < {self.convergence_threshold})"
                break
        
        # 方法2: 如果方法1没找到，使用梯度下降率判断
        if not convergence_info['converged']:
            # 计算损失下降率
            # 将数据分成多个区间，计算每个区间的平均下降率
            num_segments = 10
            segment_size = len(values) // num_segments
            
            segment_means = []
            segment_steps = []
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(values)
                segment_means.append(np.mean(values[start_idx:end_idx]))
                segment_steps.append(steps[start_idx + (end_idx - start_idx) // 2])
            
            # 计算相邻区间的下降率
            drop_rates = []
            for i in range(len(segment_means) - 1):
                drop_rate = (segment_means[i] - segment_means[i+1]) / (segment_means[i] + 1e-8)
                drop_rates.append(drop_rate)
            
            # 如果最后几个区间的下降率都很小，认为可能收敛
            if len(drop_rates) >= 3:
                recent_drop_rates = drop_rates[-3:]
                avg_recent_drop = np.mean(recent_drop_rates)
                
                if avg_recent_drop < 0.05:  # 平均下降率小于5%
                    convergence_info['converged'] = True
                    convergence_info['convergence_step'] = int(segment_steps[-3])
                    convergence_info['convergence_loss'] = float(segment_means[-1])
                    convergence_info['reason'] = f"在step {segment_steps[-3]}附近损失下降率显著降低（平均下降率: {avg_recent_drop:.4f} < 0.05)"
        
        # 方法3: 如果loss已经很低，使用更宽松的收敛判断
        if not convergence_info['converged']:
            # 检查最后20%的数据
            last_20_percent_start = int(len(values) * 0.8)
            last_20_values = values[last_20_percent_start:]
            last_20_steps = steps[last_20_percent_start:]
            last_20_ma = ma_values[last_20_percent_start:]
            
            if len(last_20_values) > 100:
                # 计算最后20%的平均loss和标准差
                mean_loss = np.mean(last_20_ma)
                std_loss = np.std(last_20_ma)
                cv = std_loss / (mean_loss + 1e-8)  # 变异系数
                
                # 如果loss已经很低（<0.01）且变异系数不太大（<0.5），认为基本收敛
                if mean_loss < 0.01 and cv < 0.5:
                    convergence_info['converged'] = True
                    convergence_info['convergence_step'] = int(last_20_steps[0])
                    convergence_info['convergence_loss'] = float(mean_loss)
                    convergence_info['reason'] = f"在step {last_20_steps[0]}附近loss已降至较低水平（平均loss: {mean_loss:.6f} < 0.01, CV: {cv:.4f} < 0.5）"
        
        # 方法4: 检测loss平台期（连续多个区间loss变化很小）
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
            
            # 从后往前找连续的平台期
            for i in range(len(segment_means) - 2, max(0, len(segment_means) - 6), -1):
                # 检查连续3个区间的变化
                recent_segments = segment_means[i:i+3]
                if len(recent_segments) == 3:
                    max_change = np.max(recent_segments) - np.min(recent_segments)
                    mean_value = np.mean(recent_segments)
                    relative_change = max_change / (mean_value + 1e-8)
                    
                    # 如果相对变化小于10%，认为进入平台期
                    if relative_change < 0.1:
                        convergence_info['converged'] = True
                        convergence_info['convergence_step'] = int(segment_steps[i])
                        convergence_info['convergence_loss'] = float(mean_value)
                        convergence_info['reason'] = f"在step {segment_steps[i]}附近进入平台期（相对变化: {relative_change:.4f} < 0.1）"
                        break
        
        # 如果还是没找到，标记为未收敛
        if not convergence_info['converged']:
            convergence_info['reason'] = "未找到持续稳定区间（可能仍在下降或波动较大）"
        
        # 计算关键统计信息
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
        """生成可视化图表"""
        # 创建简单的loss曲线图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 计算移动平均
        ma_values = self.compute_moving_average(values, self.window_size)
        
        # 绘制原始loss和移动平均
        ax.plot(steps, values, alpha=0.3, color='blue', label='Raw Loss', linewidth=0.5)
        ax.plot(steps, ma_values, color='red', label=f'Moving Average (window={self.window_size})', linewidth=2)
        
        # 标记收敛点
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
        ax.set_yscale('log')  # 使用对数刻度
        
        # 保存图片
        output_path = self.ckpt_dir / 'loss_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        plt.close()
        
        # 保存原始数据为CSV
        data_df = pd.DataFrame({
            'step': steps,
            'loss': values
        })
        csv_path = self.ckpt_dir / 'loss_data.csv'
        data_df.to_csv(csv_path, index=False)
        print(f"Saved data to: {csv_path}")
    
    def generate_report(self, convergence_info: Dict, tag: str = "train/loss") -> str:
        """生成文本报告"""
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
        """运行完整分析流程"""
        print(f"Starting loss convergence analysis...")
        print(f"Log directory: {self.log_dir}")
        print(f"Output directory: {self.ckpt_dir}")
        print(f"Tag: {tag}")
        print("")
        
        # 加载数据
        steps, values = self.load_tensorboard_data(tag)
        
        # 检测收敛
        print("\nDetecting convergence...")
        convergence_info = self.detect_convergence(steps, values)
        
        # 生成可视化
        print("\nGenerating plots...")
        self.plot_analysis(steps, values, convergence_info, tag)
        
        # 生成报告
        print("\nGenerating report...")
        report = self.generate_report(convergence_info, tag)
        
        # 保存报告
        report_path = self.ckpt_dir / 'loss_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Saved report to: {report_path}")
        
        # 打印报告
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
    
    # 创建分析器
    analyzer = LossConvergenceAnalyzer(
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        window_size=args.window_size,
        convergence_threshold=args.convergence_threshold,
        min_stable_steps=args.min_stable_steps
    )
    
    # 运行分析
    analyzer.run_analysis(tag=args.tag)


if __name__ == '__main__':
    main()
