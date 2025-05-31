from ultralytics import YOLO
from pathlib import Path
import yaml

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
        'path': str(Path.cwd() / 'data' / 'train'),  # Point to train folder
        'train': 'images',  # Images subfolder
        'val': 'images',    # Using same data for validation
        'nc': len(classes),
        'names': [classes[i] for i in range(len(classes))]
    }
    
    # Save dataset config
    config_path = Path('data/dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    return config_path

def setup_yolo_structure():
    """Create proper YOLO folder structure"""
    # Create directories
    train_images = Path('data/train/images')
    train_labels = Path('data/train/labels')
    
    train_images.mkdir(parents=True, exist_ok=True)
    
    # Move images from 'good' to 'images' folder
    good_folder = Path('data/train/good')
    if good_folder.exists():
        import shutil
        
        # Copy all images to the new structure
        for img_file in good_folder.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dest_file = train_images / img_file.name
                if not dest_file.exists():
                    shutil.copy2(img_file, dest_file)
                    print(f"Copied {img_file.name} to images folder")
    
    print(f"YOLO structure ready:")
    print(f"  Images: {train_images} ({len(list(train_images.glob('*')))} files)")
    print(f"  Labels: {train_labels} ({len(list(train_labels.glob('*.txt')))} files)")

def train_model():
    """Train a new YOLO model using transfer learning"""
    # Setup proper YOLO folder structure
    setup_yolo_structure()
    
    # Create dataset configuration
    dataset_config = create_dataset_yaml()
    
    # Load pretrained model
    model = YOLO('models/yolo8n.pt')
    
    # Train the model
    results = model.train(
        data=str(dataset_config),
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda',  # Change to 'cuda' if you have GPU
        save=True,
        project='runs/train',
        name='dart_detector',
        patience=10,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        verbose=True
    )
    
    print(f"Training completed! Best model saved at: {results.save_dir}")
    return results

if __name__ == "__main__":
    train_model()
