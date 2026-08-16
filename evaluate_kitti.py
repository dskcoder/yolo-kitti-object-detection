"""KITTI-style 2D detection evaluation with IoU and difficulty levels.

Usage:
    python evaluate_kitti.py --predictions runs/inference_output/kitti_format \
                             --ground-truth data/kitti/labels/val
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KITTIEvaluator:
    """Evaluate predictions using project-required KITTI protocol elements."""

    CORE_CLASSES = ['Car', 'Pedestrian', 'Cyclist']

    # Required IoU thresholds from project description.
    IOU_THRESHOLDS = {
        'Car': 0.7,
        'Pedestrian': 0.5,
        'Cyclist': 0.5,
        # Optional aliases to keep compatibility with broader KITTI labels.
        'Van': 0.7,
        'Truck': 0.7,
        'Person_sitting': 0.5,
        'Tram': 0.5,
        'Misc': 0.5,
    }

    DIFFICULTY = {
        'Easy': {
            'min_height': 40,
            'max_occluded': 0,
            'max_truncated': 0.15,
        },
        'Moderate': {
            'min_height': 25,
            'max_occluded': 1,
            'max_truncated': 0.30,
        },
        'Hard': {
            'min_height': 25,
            'max_occluded': 2,
            'max_truncated': 0.50,
        },
    }
    
    @staticmethod
    def parse_annotation_file(file_path):
        """Parse KITTI annotation file."""
        annotations = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            x1, y1, x2, y2 = [float(x) for x in parts[4:8]]
            annotation = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [x1, y1, x2, y2],
                'height': max(0.0, y2 - y1),
            }

            # Predicted KITTI text lines contain confidence as last token.
            if len(parts) >= 9:
                annotation['confidence'] = float(parts[-1])

            annotations.append(annotation)

        return annotations
    
    @staticmethod
    def compute_iou(bbox1, bbox2):
        """Compute IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Compute intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Compute union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

    @staticmethod
    def intersection_over_detection(det_bbox, region_bbox):
        """Compute intersection area normalized by detection area."""
        x1_min, y1_min, x1_max, y1_max = det_bbox
        x2_min, y2_min, x2_max, y2_max = region_bbox

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        det_area = max((x1_max - x1_min) * (y1_max - y1_min), 1e-12)
        return inter_area / det_area

    @staticmethod
    def matches_difficulty(annotation, difficulty_name):
        """Check whether a GT annotation satisfies KITTI difficulty constraints."""
        criteria = KITTIEvaluator.DIFFICULTY[difficulty_name]
        return (
            annotation.get('height', 0.0) >= criteria['min_height']
            and annotation.get('occluded', 3) <= criteria['max_occluded']
            and annotation.get('truncated', 1.0) <= criteria['max_truncated']
        )

    @staticmethod
    def precision_recall(tp, fp, num_gt):
        """Compute precision-recall arrays from sorted TP/FP vectors."""
        if num_gt == 0 or len(tp) == 0:
            return np.array([]), np.array([])

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / max(num_gt, 1)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
        return precisions, recalls
    
    def compute_average_precision(self, precisions, recalls):
        """Compute AP using integral interpolation of PR curve."""
        if len(precisions) == 0 or len(recalls) == 0:
            return 0.0

        # Append sentinel values
        recalls = np.concatenate([[0], recalls, [1]])
        precisions = np.concatenate([[1], precisions, [0]])

        # Compute area under curve
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

        return ap

    def evaluate_class_difficulty(self, pred_files, gt_files, class_name, difficulty_name):
        """Evaluate one class at one KITTI difficulty level."""
        iou_threshold = self.IOU_THRESHOLDS[class_name]

        gt_by_image = {}
        ignored_gt_by_image = {}
        dontcare_by_image = {}
        gt_matched = {}
        predictions = []
        num_gt = 0

        pred_lookup = {p.stem: p for p in pred_files}
        gt_files = sorted(Path(gt_files).glob('*.txt'))

        for gt_file in gt_files:
            image_id = gt_file.stem
            gt_annotations = self.parse_annotation_file(gt_file)

            class_gt = [
                ann for ann in gt_annotations
                if ann['type'] == class_name and self.matches_difficulty(ann, difficulty_name)
            ]
            ignored_gt = [
                ann for ann in gt_annotations
                if ann['type'] == class_name and not self.matches_difficulty(ann, difficulty_name)
            ]
            dontcare = [ann for ann in gt_annotations if ann['type'] == 'DontCare']

            gt_by_image[image_id] = class_gt
            ignored_gt_by_image[image_id] = ignored_gt
            dontcare_by_image[image_id] = dontcare
            gt_matched[image_id] = [False] * len(class_gt)
            num_gt += len(class_gt)

            pred_file = pred_lookup.get(image_id)
            if pred_file:
                pred_annotations = self.parse_annotation_file(pred_file)
                for pred in pred_annotations:
                    if pred['type'] == class_name:
                        predictions.append({
                            'image_id': image_id,
                            'bbox': pred['bbox'],
                            'confidence': pred.get('confidence', 1.0),
                        })

        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        tp_list = []
        fp_list = []

        for pred in predictions:
            image_id = pred['image_id']
            pred_bbox = pred['bbox']

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_by_image.get(image_id, [])):
                if gt_matched[image_id][gt_idx]:
                    continue
                iou = self.compute_iou(pred_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                gt_matched[image_id][best_gt_idx] = True
            else:
                # Ignore detections on same-class GT that fall outside this
                # difficulty tier, and detections overlapping DontCare regions.
                ignored_match = any(
                    self.compute_iou(pred_bbox, ann['bbox']) >= iou_threshold
                    for ann in ignored_gt_by_image.get(image_id, [])
                )
                dontcare_match = any(
                    self.intersection_over_detection(pred_bbox, ann['bbox']) >= 0.5
                    for ann in dontcare_by_image.get(image_id, [])
                )

                if ignored_match or dontcare_match:
                    continue

                tp_list.append(0)
                fp_list.append(1)

        tp = np.array(tp_list, dtype=int)
        fp = np.array(fp_list, dtype=int)

        precisions, recalls = self.precision_recall(tp, fp, num_gt)
        ap = self.compute_average_precision(precisions, recalls)

        # Final operating-point style precision/recall from full list.
        num_tp = int(tp.sum())
        num_fp = int(fp.sum())
        precision = (num_tp / max(num_tp + num_fp, 1)) if len(predictions) else 0.0
        recall = (num_tp / max(num_gt, 1)) if num_gt else 0.0

        return {
            'class': class_name,
            'difficulty': difficulty_name,
            'iou_threshold': iou_threshold,
            'ap': float(ap),
            'precision': float(precision),
            'recall': float(recall),
            'num_gt': int(num_gt),
            'num_predictions': int(len(tp)),
            'num_tp': int(num_tp),
            'num_fp': int(num_fp),
            'pr_curve': {
                'precision': [float(x) for x in precisions.tolist()],
                'recall': [float(x) for x in recalls.tolist()],
            },
        }

    def evaluate(self, pred_dir, gt_dir):
        """Evaluate core classes across KITTI difficulty levels."""
        pred_dir = Path(pred_dir)
        gt_dir = Path(gt_dir)

        pred_files = sorted(pred_dir.glob('*.txt'))

        logger.info(f"Evaluating {len(pred_files)} predictions...")

        results = {}
        for class_name in self.CORE_CLASSES:
            results[class_name] = {}
            for difficulty_name in self.DIFFICULTY.keys():
                logger.info(
                    f"Evaluating {class_name} | {difficulty_name} "
                    f"(IoU {self.IOU_THRESHOLDS[class_name]:.2f})"
                )
                results[class_name][difficulty_name] = self.evaluate_class_difficulty(
                    pred_files,
                    gt_dir,
                    class_name,
                    difficulty_name,
                )

        return results


def summarize_metrics(results):
    """Build project-focused summary metrics."""
    classes = ['Car', 'Pedestrian', 'Cyclist']
    difficulties = ['Easy', 'Moderate', 'Hard']

    summary = {}
    for difficulty in difficulties:
        aps = [results[c][difficulty]['ap'] for c in classes]
        summary[f'mAP_{difficulty.lower()}'] = float(np.mean(aps)) if aps else 0.0

    summary['mAP_moderate_core'] = summary['mAP_moderate']
    return summary


def main():
    """Main evaluation entry point."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO predictions on KITTI using official protocol'
    )
    
    parser.add_argument('--predictions', type=str, required=True,
                        help='Directory containing prediction files')
    parser.add_argument('--ground-truth', type=str, required=True,
                        help='Directory containing ground truth annotation files')
    parser.add_argument('--output', type=str, default='runs/evaluation_results',
                        help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🔍 Starting KITTI Evaluation")
    logger.info("=" * 80)
    
    # Run evaluation
    evaluator = KITTIEvaluator()
    results = evaluator.evaluate(args.predictions, args.ground_truth)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("📊 Evaluation Results")
    logger.info("=" * 80)
    
    difficulties = ['Easy', 'Moderate', 'Hard']
    for class_name in KITTIEvaluator.CORE_CLASSES:
        logger.info(f"\n{class_name}:")
        for difficulty in difficulties:
            result = results[class_name][difficulty]
            logger.info(
                f"  {difficulty:<8} AP={result['ap']:.4f} "
                f"P={result['precision']:.4f} R={result['recall']:.4f} "
                f"GT={result['num_gt']} Pred={result['num_predictions']}"
            )

    summary = summarize_metrics(results)
    logger.info(f"\n{'='*40}")
    logger.info(f"mAP Easy (Car/Pedestrian/Cyclist): {summary['mAP_easy']:.4f}")
    logger.info(f"mAP Moderate (Car/Pedestrian/Cyclist): {summary['mAP_moderate']:.4f}")
    logger.info(f"mAP Hard (Car/Pedestrian/Cyclist): {summary['mAP_hard']:.4f}")
    logger.info(f"{'='*40}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'protocol': {
                'classes': KITTIEvaluator.CORE_CLASSES,
                'iou_thresholds': {
                    c: KITTIEvaluator.IOU_THRESHOLDS[c] for c in KITTIEvaluator.CORE_CLASSES
                },
                'difficulty_criteria': KITTIEvaluator.DIFFICULTY,
            },
            'summary': summary,
            # Backward-compatible key names used by analyzer/report scripts.
            'mAP_all_classes': summary['mAP_moderate_core'],
            'mAP_core_classes': summary['mAP_moderate_core'],
            'results_by_class_difficulty': results,
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
