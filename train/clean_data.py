#!/usr/bin/env python3
"""
YOLO Dataset Cleaner using YOLOv11m
Removes unlabeled images that contain objects detected by YOLOv11m
"""

import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from ultralytics import YOLO
import argparse


class YOLODatasetCleaner:
    def __init__(self, data_root, output_root, vis_root, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the dataset cleaner
        
        Args:
            data_root: Root directory of the dataset (contains train/val folders)
            output_root: Output directory for cleaned dataset
            vis_root: Output directory for visualizations
            conf_threshold: Confidence threshold for detection
            iou_threshold: IOU threshold for NMS
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.vis_root = Path(vis_root)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO v11m model
        print("Loading YOLOv11m model...")
        self.model = YOLO('yolo11m.pt')
        
        # Statistics
        self.stats = {
            'total_labeled': 0,
            'total_unlabeled': 0,
            'unlabeled_with_objects': 0,
            'unlabeled_clean': 0,
            'kept_images': 0
        }
        
    def load_dataset_structure(self, yaml_path):
        """Load dataset structure from YAML file"""
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        return data_config
    
    def get_image_label_pairs(self, split_dir):
        """
        Get all images and their corresponding label files
        
        Args:
            split_dir: Directory containing 'images' and 'labels' subdirectories
            
        Returns:
            labeled_images: List of image paths that have labels
            unlabeled_images: List of image paths that don't have labels
        """
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            return [], []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f'*{ext}')))
            image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
        
        labeled_images = []
        unlabeled_images = []
        
        for img_path in image_files:
            # Corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists() and label_path.stat().st_size > 0:
                labeled_images.append(img_path)
                self.stats['total_labeled'] += 1
            else:
                unlabeled_images.append(img_path)
                self.stats['total_unlabeled'] += 1
        
        return labeled_images, unlabeled_images
    
    def detect_objects(self, image_path):
        """
        Run YOLOv11m detection on an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            results: Detection results
            has_objects: Boolean indicating if any objects were detected
        """
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        has_objects = len(results.boxes) > 0
        return results, has_objects
    
    def visualize_detections(self, image_path, results, output_path):
        """
        Visualize detections on the image
        
        Args:
            image_path: Path to the original image
            results: Detection results from YOLO
            output_path: Path to save the visualization
        """
        # Read image
        img = cv2.imread(str(image_path))
        
        # Draw detections
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Get class name
            class_name = results.names[cls]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
        
        # Save visualization
        cv2.imwrite(str(output_path), img)
    
    def copy_labeled_data(self, split_name, labeled_images):
        """
        Copy all labeled data to output directory
        
        Args:
            split_name: Name of the split (train/val)
            labeled_images: List of labeled image paths
        """
        print(f"\nCopying labeled data for {split_name}...")
        
        output_images_dir = self.output_root / split_name / 'images'
        output_labels_dir = self.output_root / split_name / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(labeled_images, desc=f"Copying {split_name} labeled"):
            # Copy image
            shutil.copy2(img_path, output_images_dir / img_path.name)
            
            # Copy label
            label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, output_labels_dir / label_path.name)
            
            self.stats['kept_images'] += 1
    
    def process_unlabeled_images(self, split_name, unlabeled_images):
        """
        Process unlabeled images: detect objects and keep only clean ones
        
        Args:
            split_name: Name of the split (train/val)
            unlabeled_images: List of unlabeled image paths
        """
        print(f"\nProcessing unlabeled images for {split_name}...")
        
        output_images_dir = self.output_root / split_name / 'images'
        vis_dir = self.vis_root / split_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(unlabeled_images, desc=f"Processing {split_name} unlabeled"):
            # Detect objects
            results, has_objects = self.detect_objects(img_path)
            
            if has_objects:
                # Save visualization
                vis_path = vis_dir / img_path.name
                self.visualize_detections(img_path, results, vis_path)
                self.stats['unlabeled_with_objects'] += 1
            else:
                # Keep clean image (no objects detected)
                shutil.copy2(img_path, output_images_dir / img_path.name)
                self.stats['unlabeled_clean'] += 1
                self.stats['kept_images'] += 1
    
    def clean_dataset(self):
        """Main function to clean the dataset"""
        print("="*60)
        print("YOLO Dataset Cleaner using YOLOv11m")
        print("="*60)
        print(f"Input dataset: {self.data_root}")
        print(f"Output dataset: {self.output_root}")
        print(f"Visualizations: {self.vis_root}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IOU threshold: {self.iou_threshold}")
        print("="*60)
        
        # Load dataset configuration
        yaml_path = self.data_root / 'data.yaml'
        if not yaml_path.exists():
            print(f"Error: data.yaml not found at {yaml_path}")
            return
        
        data_config = self.load_dataset_structure(yaml_path)
        
        # Process each split (train, val)
        splits = ['train', 'val']
        
        for split in splits:
            split_dir = self.data_root / split
            if not split_dir.exists():
                print(f"Warning: {split} directory not found, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing {split.upper()} split")
            print(f"{'='*60}")
            
            # Get labeled and unlabeled images
            labeled_images, unlabeled_images = self.get_image_label_pairs(split_dir)
            
            print(f"Found {len(labeled_images)} labeled images")
            print(f"Found {len(unlabeled_images)} unlabeled images")
            
            # Copy all labeled data
            if labeled_images:
                self.copy_labeled_data(split, labeled_images)
            
            # Process unlabeled images
            if unlabeled_images:
                self.process_unlabeled_images(split, unlabeled_images)
        
        # Copy data.yaml to output directory
        output_yaml = self.output_root / 'data.yaml'
        shutil.copy2(yaml_path, output_yaml)
        
        # Update paths in data.yaml if needed
        self.update_yaml_paths(output_yaml)
        
        # Print statistics
        self.print_statistics()
    
    def update_yaml_paths(self, yaml_path):
        """Update paths in data.yaml to point to the new location"""
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths to be relative to the yaml file location
        if 'train' in data_config:
            data_config['train'] = 'train/images'
        if 'val' in data_config:
            data_config['val'] = 'val/images'
        if 'test' in data_config:
            data_config['test'] = 'test/images'
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    
    def print_statistics(self):
        """Print cleaning statistics"""
        print("\n" + "="*60)
        print("CLEANING STATISTICS")
        print("="*60)
        print(f"Total labeled images: {self.stats['total_labeled']}")
        print(f"Total unlabeled images: {self.stats['total_unlabeled']}")
        print(f"  - Unlabeled with objects detected: {self.stats['unlabeled_with_objects']}")
        print(f"  - Unlabeled clean (no objects): {self.stats['unlabeled_clean']}")
        print(f"\nTotal images kept in clean dataset: {self.stats['kept_images']}")
        print(f"Total images removed: {self.stats['unlabeled_with_objects']}")
        
        if self.stats['total_unlabeled'] > 0:
            removal_rate = (self.stats['unlabeled_with_objects'] / self.stats['total_unlabeled']) * 100
            print(f"Removal rate from unlabeled: {removal_rate:.2f}%")
        
        print("="*60)
        print("\nCleaning completed successfully!")
        print(f"Clean dataset saved to: {self.output_root}")
        print(f"Visualizations saved to: {self.vis_root}")


def main():
    parser = argparse.ArgumentParser(
        description='Clean YOLO dataset by removing unlabeled images with objects detected by YOLOv11m'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/yoloworld_v3',
        help='Root directory of the input dataset'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='data/yoloworld_v3_clean',
        help='Output directory for cleaned dataset'
    )
    parser.add_argument(
        '--vis_root',
        type=str,
        default='data/yoloworld_v3_clean_vis',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detection (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS (default: 0.45)'
    )
    
    args = parser.parse_args()
    
    # Create cleaner instance
    cleaner = YOLODatasetCleaner(
        data_root=args.data_root,
        output_root=args.output_root,
        vis_root=args.vis_root,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run cleaning
    cleaner.clean_dataset()


if __name__ == '__main__':
    main()