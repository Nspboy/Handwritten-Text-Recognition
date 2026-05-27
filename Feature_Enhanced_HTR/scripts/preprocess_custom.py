"""
Preprocess Custom Images and Prepare for Training.
This script takes images from 'dataset/custom_images', applies the preprocessing pipeline,
saves the 'modified' images, and creates a template for labeling.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import json

# Add root to path so we can import modules
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from engine.preprocessing.preprocess import ImagePreprocessor

def main():
    preprocessor = ImagePreprocessor()
    
    custom_dir = root_dir / "data" / "custom"
    output_dir = root_dir / "data" / "processed" / "custom"
    label_file = root_dir / "data" / "labels" / "custom_labels.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing labels
    existing_labels = {}
    if label_file.exists():
        with open(label_file, "r") as f:
            data = json.load(f)
            for item in data:
                existing_labels[item["image"]] = item["text"]
    
    new_labels = []
    
    print(f"--- Processing images from {custom_dir} ---")
    image_files = list(custom_dir.glob("*.png")) + list(custom_dir.glob("*.jpg"))
    
    if not image_files:
        print("No images found in dataset/custom_images")
        return

    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        
        # 1. Load and Preprocess
        processed = preprocessor.preprocess_image(str(img_path))
        
        if processed is not None:
            # 2. Save modified image
            save_path = output_dir / img_path.name
            cv2.imwrite(str(save_path), processed)
            print(f"  [OK] Saved modified image to {save_path.relative_to(root_dir)}")
            
            # 3. Add to label list
            text = existing_labels.get(img_path.name, "INSERT_TEXT_HERE")
            new_labels.append({
                "image": img_path.name,
                "text": text,
                "processed_path": str(save_path.relative_to(root_dir))
            })
        else:
            print(f"  [FAIL] Failed to process {img_path.name}")

    # Save label template
    with open(label_file, "w") as f:
        json.dump(new_labels, f, indent=2)
    
    print("\n--- Done ---")
    print(f"1. Modified images are in: {output_dir.relative_to(root_dir)}")
    print(f"2. Label template created at: {label_file.relative_to(root_dir)}")
    print("   Please open the JSON file and replace 'INSERT_TEXT_HERE' with the actual text in the images.")
    print("   Then run the training pipeline with these labels.")

if __name__ == "__main__":
    main()
