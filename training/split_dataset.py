#!/usr/bin/env python3
"""
Script to split YOLO training dataset into training and validation sets.
This script will move 20% of the data from train/ to val/ directories.
"""

import os
import random
import shutil
from pathlib import Path

def split_dataset(train_dir, val_dir, validation_split=0.2):
    """
    Split dataset into training and validation sets.
    
    Args:
        train_dir (str): Path to training directory containing images/ and labels/
        val_dir (str): Path to validation directory (will be created if not exists)
        validation_split (float): Fraction of data to use for validation (default: 0.2 = 20%)
    """
    
    train_images_dir = Path(train_dir) / "images"
    train_labels_dir = Path(train_dir) / "labels"
    val_images_dir = Path(val_dir) / "images"
    val_labels_dir = Path(val_dir) / "labels"
    
    # Ensure validation directories exist
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(train_images_dir.glob("*.jpg"))
    
    # Filter to only include images that have corresponding label files
    valid_files = []
    for img_file in image_files:
        label_file = train_labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_files.append(img_file.stem)
    
    print(f"Found {len(valid_files)} valid image-label pairs")
    
    # Shuffle the files for random split
    random.seed(42)  # For reproducible results
    random.shuffle(valid_files)
    
    # Calculate split point
    val_count = int(len(valid_files) * validation_split)
    train_count = len(valid_files) - val_count
    
    print(f"Splitting into:")
    print(f"  Training: {train_count} samples ({(1-validation_split)*100:.1f}%)")
    print(f"  Validation: {val_count} samples ({validation_split*100:.1f}%)")
    
    # Get validation files (last val_count files after shuffle)
    val_files = valid_files[-val_count:]
    
    # Move validation files
    moved_images = 0
    moved_labels = 0
    
    for file_stem in val_files:
        # Move image file
        src_img = train_images_dir / f"{file_stem}.jpg"
        dst_img = val_images_dir / f"{file_stem}.jpg"
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
            moved_images += 1
        
        # Move label file
        src_label = train_labels_dir / f"{file_stem}.txt"
        dst_label = val_labels_dir / f"{file_stem}.txt"
        if src_label.exists():
            shutil.move(str(src_label), str(dst_label))
            moved_labels += 1
    
    print(f"\nMoved {moved_images} images and {moved_labels} labels to validation set")
    
    # Verify final counts
    final_train_images = len(list(train_images_dir.glob("*.jpg")))
    final_train_labels = len(list(train_labels_dir.glob("*.txt")))
    final_val_images = len(list(val_images_dir.glob("*.jpg")))
    final_val_labels = len(list(val_labels_dir.glob("*.txt")))
    
    print(f"\nFinal dataset distribution:")
    print(f"  Training: {final_train_images} images, {final_train_labels} labels")
    print(f"  Validation: {final_val_images} images, {final_val_labels} labels")
    print(f"  Total: {final_train_images + final_val_images} images, {final_train_labels + final_val_labels} labels")

if __name__ == "__main__":
    # Set paths
    base_dir = Path(__file__).parent / "data"
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    
    print("YOLO Dataset Splitter")
    print("=" * 50)
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print()
    
    # Check if directories exist
    if not train_dir.exists():
        print(f"Error: Training directory {train_dir} does not exist!")
        exit(1)
    
    if not (train_dir / "images").exists() or not (train_dir / "labels").exists():
        print(f"Error: Training directory must contain 'images' and 'labels' subdirectories!")
        exit(1)
    
    # Perform the split
    split_dataset(str(train_dir), str(val_dir), validation_split=0.2)
    
    print("\nDataset split completed successfully!")
    print("You can now use these directories for YOLO training:")
    print(f"  train: {train_dir}")
    print(f"  val: {val_dir}")
