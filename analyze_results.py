"""
YOLO KITTI Analysis Utilities
Post-training analysis and visualization tools

Usage:
    python analyze_results.py --results runs/detect/yolo11m_kitti_YYYYMMDD_HHMMSS
"""

import argparse
import json
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze YOLO training and inference results."""
    
    def __init__(self, results_dir):
        """Initialize analyzer."""
        self.results_dir = Path(results_dir)
        logger.info(f"Analyzing results from: {self.results_dir}")
    
    def load_training_results(self):
        """Load training results from CSV."""
        results_csv = self.results_dir / 'results.csv'
        
        if not results_csv.exists():
            logger.warning(f"Results CSV not found: {results_csv}")
            return None
        
        results = []
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        
        return results
    
    def load_evaluation_metrics(self, eval_file):
        """Load evaluation metrics from JSON."""
        with open(eval_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    
    def plot_training_curves(self, output_dir='results'):
        """Plot training metrics over time."""
        results = self.load_training_results()
        
        if results is None:
            logger.warning("Could not load training results")
            return
        
        # Extract metrics (current Ultralytics CSV columns)
        epochs = []
        train_box_loss = []
        train_cls_loss = []
        train_dfl_loss = []
        val_box_loss = []
        val_cls_loss = []
        val_dfl_loss = []
        map50 = []
        map5095 = []
        
        for result in results:
            try:
                epochs.append(int(float(result.get('epoch', 0))))
                train_box_loss.append(float(result.get('train/box_loss', 0)))
                train_cls_loss.append(float(result.get('train/cls_loss', 0)))
                train_dfl_loss.append(float(result.get('train/dfl_loss', 0)))
                val_box_loss.append(float(result.get('val/box_loss', 0)))
                val_cls_loss.append(float(result.get('val/cls_loss', 0)))
                val_dfl_loss.append(float(result.get('val/dfl_loss', 0)))
                map50.append(float(result.get('metrics/mAP50(B)', 0)))
                map5095.append(float(result.get('metrics/mAP50-95(B)', 0)))
            except (ValueError, TypeError):
                continue
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(epochs, train_box_loss, label='Train Box', marker='o')
        axes[0].plot(epochs, train_cls_loss, label='Train Cls', marker='o')
        axes[0].plot(epochs, train_dfl_loss, label='Train DFL', marker='o')
        axes[0].plot(epochs, val_box_loss, label='Val Box', marker='s')
        axes[0].plot(epochs, val_cls_loss, label='Val Cls', marker='s')
        axes[0].plot(epochs, val_dfl_loss, label='Val DFL', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss Components')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # mAP plot
        axes[1].plot(epochs, map50, label='mAP50', marker='o', color='green')
        axes[1].plot(epochs, map5095, label='mAP50-95', marker='o', color='blue')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP')
        axes[1].set_title('Validation mAP Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Training curves saved to {output_dir / 'training_curves.png'}")
        plt.close()
    
    def plot_class_performance(self, eval_metrics, output_dir='results'):
        """Plot per-class AP performance."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        classes = []
        aps = []

        # New format: results_by_class_difficulty[class][difficulty].
        by_class_diff = eval_metrics.get('results_by_class_difficulty', {})
        if by_class_diff:
            for class_name, diff_data in by_class_diff.items():
                moderate = diff_data.get('Moderate')
                if not moderate:
                    continue
                classes.append(class_name)
                aps.append(moderate.get('ap', 0) * 100)
        else:
            # Backward compatibility for old format.
            for class_name, metrics in eval_metrics.get('results_by_class', {}).items():
                classes.append(class_name)
                aps.append(metrics.get('ap', 0) * 100)
        
        if not classes:
            logger.warning("No class performance data found")
            return
        
        # Plot bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(classes, aps, color='steelblue', alpha=0.8)
        ax.set_ylabel('AP (%)', fontsize=12)
        ax.set_title('Per-Class AP Performance', fontsize=14)
        ax.set_ylim([0, 100])
        
        # Add value labels on bars
        for bar, ap in zip(bars, aps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ap:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Class performance plot saved to {output_dir / 'class_performance.png'}")
        plt.close()
    
    def generate_summary_report(self, eval_file, output_dir='results'):
        """Generate summary report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        eval_metrics = self.load_evaluation_metrics(eval_file)
        
        report = []
        report.append("=" * 80)
        report.append("YOLO KITTI Object Detection - Results Summary")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        summary = eval_metrics.get('summary', {})
        if summary:
            report.append(f"mAP Easy (Car/Pedestrian/Cyclist): {summary.get('mAP_easy', 0)*100:.2f}%")
            report.append(f"mAP Moderate (Car/Pedestrian/Cyclist): {summary.get('mAP_moderate', 0)*100:.2f}%")
            report.append(f"mAP Hard (Car/Pedestrian/Cyclist): {summary.get('mAP_hard', 0)*100:.2f}%")
        else:
            mAP_all = eval_metrics.get('mAP_all_classes', 0)
            mAP_core = eval_metrics.get('mAP_core_classes', 0)
            report.append(f"Mean Average Precision (all classes): {mAP_all*100:.2f}%")
            report.append(f"Mean Average Precision (Car/Pedestrian/Cyclist): {mAP_core*100:.2f}%")
        report.append("")
        
        # Per-class metrics (Moderate by default for comparison)
        report.append("Per-Class Performance:")
        report.append("-" * 80)
        report.append(f"{'Class':<20} {'Difficulty':<15} {'AP':<12} {'Detections':<12} {'GT':<12}")
        report.append("-" * 80)

        by_class_diff = eval_metrics.get('results_by_class_difficulty', {})
        if by_class_diff:
            for class_name, diff_data in by_class_diff.items():
                for difficulty_name in ('Easy', 'Moderate', 'Hard'):
                    metrics = diff_data.get(difficulty_name)
                    if not metrics:
                        continue
                    report.append(
                        f"{class_name:<20} {difficulty_name:<15} {metrics.get('ap', 0)*100:>6.2f}%       "
                        f"{metrics.get('num_predictions', 0):>6}      {metrics.get('num_gt', 0):>6}"
                    )
        else:
            for class_name, metrics in eval_metrics.get('results_by_class', {}).items():
                report.append(
                    f"{class_name:<20} {'Moderate':<15} {metrics.get('ap', 0)*100:>6.2f}%       "
                    f"{metrics.get('num_predictions', 0):>6}      {metrics.get('num_gt', 0):>6}"
                )
        
        report.append("-" * 80)
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        
        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info(f"✓ Summary report saved to {output_dir / 'summary_report.txt'}")
        print(report_text)
    
    def analyze_predictions_stats(self, predictions_file, output_dir='results'):
        """Analyze prediction statistics."""
        if not Path(predictions_file).exists():
            logger.warning(f"Predictions file not found: {predictions_file}")
            return
        
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract statistics
        class_counts = {}
        confidence_scores = []
        
        for pred_list in predictions:
            for pred in pred_list:
                class_name = pred.get('class', 'Unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidence_scores.append(pred.get('confidence', 0))
        
        # Plot class distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class count bar chart
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        axes[0].bar(classes, counts, color='skyblue', alpha=0.8)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Detection Count by Class')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Confidence score histogram
        axes[1].hist(confidence_scores, bins=20, color='lightgreen', alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Confidence Scores')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_statistics.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Prediction statistics saved")
        plt.close()


def main():
    """Main analysis entry point."""
    
    parser = argparse.ArgumentParser(
        description='Analyze YOLO KITTI object detection results'
    )
    
    parser.add_argument('--results', type=str, required=True,
                        help='Path to training results directory')
    parser.add_argument('--eval-file', type=str,
                        help='Path to evaluation results JSON file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for analysis plots')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results)
    
    logger.info("=" * 80)
    logger.info("📊 Starting Results Analysis")
    logger.info("=" * 80)
    
    # Plot training curves
    logger.info("\nPlotting training curves...")
    analyzer.plot_training_curves(args.output)
    
    # Analyze evaluation metrics
    if args.eval_file:
        logger.info("\nAnalyzing evaluation metrics...")
        eval_metrics = analyzer.load_evaluation_metrics(args.eval_file)
        
        logger.info("\nPlotting class performance...")
        analyzer.plot_class_performance(eval_metrics, args.output)
        
        logger.info("\nGenerating summary report...")
        analyzer.generate_summary_report(args.eval_file, args.output)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Analysis complete! Results saved to {args.output}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
