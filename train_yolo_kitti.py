"""
YOLO KITTI Object Detection - Training Script
Project: 2D Object Detection for Autonomous Driving
Dataset: KITTI Vision Benchmark Suite
Model: YOLO11m (Ultralytics)

Usage:
    python train_yolo_kitti.py --epochs 100 --batch 16 --device 0 --imgsz 640
"""

import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import torch
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories for the project."""
    dirs = [
        'runs/detect',
        'data/kitti',
        'data/kitti/images',
        'data/kitti/labels',
        'results',
        'visualizations'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory ready: {d}")


def create_kitti_yaml(output_path='kitti.yaml'):
    """Create KITTI dataset configuration YAML file if missing."""
    output_file = Path(output_path)
    if output_file.exists():
        logger.info(f"✓ Dataset config exists: {output_path}")
        return

    yaml_content = """# Official KITTI Object Detection Dataset (local copy)
# Update `path` if your extracted KITTI folder is in another location.
path: ./data/kitti
train: images/train
val: images/val
test: images/test

# Number of classes (KITTI has 8 object types)
nc: 8

# Class names (KITTI standard classes)
names: ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    logger.info(f"✓ Created {output_path}")


def validate_local_dataset_config(data_path: str):
    """Validate local dataset config and fail fast to avoid non-official auto-downloads."""
    config_path = Path(data_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config '{data_path}' not found. "
            "Use official KITTI from the course PDF, extract it locally, and point --data to that YAML. "
            "You can generate a template with --init-data-config."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    missing_keys = [k for k in ('path', 'train', 'val') if k not in cfg]
    if missing_keys:
        raise ValueError(f"Dataset config '{data_path}' is missing keys: {missing_keys}")

    root = Path(cfg['path'])
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    train_images = root / str(cfg['train'])
    val_images = root / str(cfg['val'])
    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(
            f"Configured dataset paths do not exist. train='{train_images}', val='{val_images}'."
        )

    train_labels = Path(str(train_images).replace('images', 'labels'))
    val_labels = Path(str(val_images).replace('images', 'labels'))
    if not train_labels.exists() or not val_labels.exists():
        raise FileNotFoundError(
            f"Expected labels directories not found. train='{train_labels}', val='{val_labels}'."
        )


def train_yolo(args):
    """Train YOLO model on KITTI dataset."""
    
    logger.info("=" * 80)
    logger.info("🚀 Starting YOLO KITTI Object Detection Training")
    logger.info("=" * 80)
    
    # Device setup
    if args.device == -1:
        device = 'cpu'
    else:
        device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Check if CUDA is available
    if device != 'cpu':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Load model
    logger.info(f"\n📦 Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Log model info
    logger.info(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Training configuration
    training_config = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': device,
        'patience': 20,  # Early stopping patience
        'cache': False,  # Cache images (set True if RAM available)
        'augment': True,  # Data augmentation
        'cos_lr': True,  # Cosine learning rate scheduler
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        # Loss weights
        'box': 7.5,  # Box loss weight
        'cls': 0.5,  # Class loss weight
        'dfl': 1.5,  # Distribution focal loss weight
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        # Save settings
        'save': True,
        'save_period': 10,
        'plots': True,
        'project': 'runs/detect',
        'name': args.exp_name,
        'exist_ok': False,
    }
    
    logger.info("\n📋 Training Configuration:")
    for key, value in training_config.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    logger.info(f"\n🎯 Starting training for {args.epochs} epochs...")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Batch size: {args.batch} | Image size: {args.imgsz}x{args.imgsz}")
    
    try:
        results = model.train(**training_config)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Training Completed Successfully!")
        logger.info("=" * 80)
        
        # Log results
        logger.info(f"Best model saved to: {results.save_dir}/weights/best.pt")
        logger.info(f"Training plots saved to: {results.save_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        raise


def validate_model(model_path, data_path='kitti.yaml'):
    """Validate trained model on KITTI validation set."""
    
    logger.info("\n" + "=" * 80)
    logger.info("🔍 Starting Model Validation")
    logger.info("=" * 80)
    
    model = YOLO(model_path)
    
    logger.info(f"Validating on: {data_path}")
    metrics = model.val(data=data_path, imgsz=640)
    
    logger.info("\n📊 Validation Results:")
    logger.info(f"  mAP50: {metrics.box.map50:.4f}")
    logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"  Precision: {metrics.box.mp:.4f}")
    logger.info(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model(model_path, export_format='onnx'):
    """Export model to different formats."""
    
    logger.info(f"\n📤 Exporting model to {export_format} format...")
    model = YOLO(model_path)
    
    export_path = model.export(format=export_format)
    logger.info(f"✓ Model exported to: {export_path}")
    
    return export_path


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='YOLO KITTI Object Detection Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_yolo_kitti.py --epochs 100 --batch 16
  python train_yolo_kitti.py --model yolo11l.pt --epochs 200 --batch 32 --device 0
  python train_yolo_kitti.py --data custom_kitti.yaml --imgsz 1024
        """
    )
    
    parser.add_argument('--model', type=str, default='yolo11m.pt',
                        help='Model to train: yolo11n, yolo11m (default), yolo11l, yolo11x')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (default: 0, use -1 for CPU)')
    parser.add_argument('--data', type=str, default='kitti.yaml',
                        help='Path to local official KITTI dataset YAML file (default: kitti.yaml)')
    parser.add_argument('--init-data-config', action='store_true',
                        help='Create a local dataset YAML template file and exit')
    parser.add_argument('--exp-name', type=str, 
                        default=f"yolo11m_kitti_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help='Experiment name for output directory')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation after training')
    parser.add_argument('--export', type=str, default=None,
                        choices=['onnx', 'torchscript', 'tflite', 'pb', 'savedmodel'],
                        help='Export model to specified format')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()

    if args.init_data_config:
        create_kitti_yaml(args.data)
        logger.info("Local dataset template created. Exiting because --init-data-config was set.")
        return

    validate_local_dataset_config(args.data)
    
    # Train
    results = train_yolo(args)
    
    # Validate if requested
    if args.validate:
        best_model = Path(results.save_dir) / 'weights' / 'best.pt'
        validate_model(str(best_model), args.data)
    
    # Export if requested
    if args.export:
        best_model = Path(results.save_dir) / 'weights' / 'best.pt'
        export_model(str(best_model), args.export)
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 All Done! Check 'runs/detect/' for results")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
