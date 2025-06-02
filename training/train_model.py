from ultralytics import YOLO
from pathlib import Path
import yaml
import os

def create_dataset_yaml():
    """Create dataset configuration file for YOLO training"""
    # Correct class mapping according to user input
    classes = {
        0: "20",
        1: "3",
        2: "11",
        3: "6",
        4: "dart",
        5: "9",
        6: "15"
    }
    
    dataset_config = {
        'path': str(Path.cwd() / 'training' / 'data' / 'train'),  # Point to train folder
        'train': 'images',  # Images subfolder
        'val': 'images',    # Using same data for validation
        'nc': len(classes),
        'names': [classes[i] for i in range(len(classes))]
    }
    
    # Save dataset config
    config_path = Path('training/data/dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    return config_path

def setup_yolo_structure():
    """Create proper YOLO folder structure"""
    # Create directories
    train_images = Path('training/data/train/images')
    train_labels = Path('training/data/train/labels')
    
    train_images.mkdir(parents=True, exist_ok=True)
    
    print(f"YOLO structure ready:")
    print(f"  Images: {train_images} ({len(list(train_images.glob('*')))} files)")
    print(f"  Labels: {train_labels} ({len(list(train_labels.glob('*.txt')))} files)")

def train_model():
    """Train a new YOLO model using transfer learning"""
    # Setup proper YOLO folder structure
    setup_yolo_structure()
    
    # Create dataset configuration
    dataset_config = create_dataset_yaml()    # Load pretrained model - Using YOLOv11 for better small object detection
    model = YOLO('models/best.pt')
    
    # Use all available CPU cores for multiprocessing
    num_workers = os.cpu_count()
    print(f"Using {num_workers} workers for multiprocessing")
      # Train the model with optimized hyperparameters for dart detection
    results = model.train(
        data=str(dataset_config),
        val=True, epochs=100, imgsz=800, workers=16, project="training/runs/train", name="DartDetector", lr0=0.00517, lrf=0.00761, momentum=0.90098, weight_decay=0.00038, warmup_epochs=1.94204, warmup_momentum=0.42999, box=2.99452, cls=0.30763, dfl=1.53753, hsv_h=0.00695, hsv_s=0.45949, hsv_v=0.24372, degrees=15.58584, translate=0.10067, scale=0.2181, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.31783, mosaic=0.61939, mixup=0.0, copy_paste=0.0, batch=-1, 
        verbose=True
    )
    
    print(f"Training completed! Best model saved at: {results.save_dir}")
    return results

if __name__ == "__main__":
    train_model()
