"""
YOLO KITTI Object Detection - Inference & Evaluation
Generate predictions and visualizations for KITTI test set

Usage:
    python inference_yolo_kitti.py --model runs/detect/yolo11m_kitti/weights/best.pt 
                                   --source data/kitti/images/val 
                                   --conf 0.5
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOInference:
    """YOLO inference class for KITTI dataset."""
    
    KITTI_CLASSES = {
        0: 'Car',
        1: 'Van',
        2: 'Truck',
        3: 'Pedestrian',
        4: 'Person_sitting',
        5: 'Cyclist',
        6: 'Tram',
        7: 'Misc'
    }
    
    def __init__(self, model_path, conf_threshold=0.5, device=0):
        """Initialize YOLO model."""
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cpu' if device == -1 else device
        logger.info(f"✓ Model loaded. Confidence threshold: {conf_threshold}")
    
    def predict(self, source, save_txt=False):
        """Run inference on source images."""
        logger.info(f"Running inference on: {source}")
        
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            device=self.device,
            save=True,
            save_txt=save_txt,
            project='runs/detect',
            name='inference_results'
        )
        
        logger.info(f"✓ Inference completed. Results saved to runs/detect/inference_results")
        return results
    
    def predict_single(self, image_path):
        """Predict on single image."""
        results = self.model.predict(image_path, conf=self.conf_threshold, device=self.device)
        return results[0]
    
    def extract_predictions(self, result):
        """Extract predictions in KITTI format."""
        predictions = []
        
        if result.boxes is None:
            return predictions
        
        # Get box coordinates and confidences
        boxes = result.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            
            prediction = {
                'class': self.KITTI_CLASSES.get(cls_id, 'Unknown'),
                'class_id': int(cls_id),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                },
                'confidence': float(conf)
            }
            predictions.append(prediction)
        
        return predictions
    
    def save_predictions_kitti_format(self, results, output_dir):
        """Save predictions in KITTI format from Ultralytics result objects."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {len(results)} images...")

        for result in results:
            predictions = self.extract_predictions(result)

            # Save in KITTI format
            image_stem = Path(result.path).stem
            output_file = Path(output_dir) / f"{image_stem}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for pred in predictions:
                    bbox = pred['bbox']
                    # KITTI format: type truncated occluded alpha bbox conf
                    # For object detection: type truncated occluded alpha x1 y1 x2 y2 conf
                    line = f"{pred['class']} -1 -1 0 {bbox['x1']:.2f} {bbox['y1']:.2f} {bbox['x2']:.2f} {bbox['y2']:.2f} {pred['confidence']:.4f}\n"
                    f.write(line)
        
        logger.info(f"✓ Predictions saved to {output_dir}")
    
    def visualize_predictions(self, image_path, save_path=None):
        """Visualize predictions on image."""
        result = self.predict_single(image_path)
        
        image = cv2.imread(str(image_path))
        predictions = self.extract_predictions(result)
        
        # Draw bounding boxes
        for pred in predictions:
            bbox = pred['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            conf = pred['confidence']
            class_name = pred['class']
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, image)
            logger.info(f"✓ Visualization saved to {save_path}")
        
        return image
    
    def create_statistics(self, predictions_list):
        """Create statistics from predictions."""
        stats = {
            'total_detections': len(predictions_list),
            'detections_by_class': {},
            'confidence_stats': {}
        }
        
        confidences = []
        
        for preds in predictions_list:
            for pred in preds:
                class_name = pred['class']
                conf = pred['confidence']
                
                stats['detections_by_class'][class_name] = \
                    stats['detections_by_class'].get(class_name, 0) + 1
                confidences.append(conf)
        
        if confidences:
            stats['confidence_stats'] = {
                'min': float(min(confidences)),
                'max': float(max(confidences)),
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences))
            }
        
        return stats


def main():
    """Main inference entry point."""
    
    parser = argparse.ArgumentParser(
        description='YOLO KITTI Inference & Evaluation'
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--source', type=str, required=True,
                        help='Image or directory path for inference')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (default: 0, use -1 for CPU)')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save predictions as text files')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--output-dir', type=str, default='runs/inference_output',
                        help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = YOLOInference(args.model, conf_threshold=args.conf, device=args.device)
    
    logger.info("=" * 80)
    logger.info("🔍 Starting YOLO KITTI Inference")
    logger.info("=" * 80)
    
    # Run inference
    results = inference.predict(
        source=args.source,
        save_txt=args.save_txt
    )
    
    # Save in KITTI format
    if Path(args.source).is_dir():
        output_kitti_dir = Path(args.output_dir) / 'kitti_format'
        inference.save_predictions_kitti_format(results, str(output_kitti_dir))
    
    # Create visualizations
    if args.visualize and Path(args.source).is_dir():
        vis_dir = Path(args.output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating visualizations...")
        image_files = list(Path(args.source).glob('*.png')) + \
                     list(Path(args.source).glob('*.jpg'))
        
        for image_path in image_files:
            save_path = vis_dir / (image_path.stem + '_vis.png')
            inference.visualize_predictions(str(image_path), str(save_path))
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Inference completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
