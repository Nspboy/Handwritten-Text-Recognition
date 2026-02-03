"""
Generate Sample Dataset for Handwritten Text Recognition

Creates synthetic handwritten text images for training and testing the HTR model.
"""

import os
import json
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleDatasetGenerator:
    """Generate synthetic handwritten-style text images."""
    
    def __init__(self, output_dir="dataset", num_samples=100):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Output directory for dataset
            num_samples: Number of samples to generate
        """
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.raw_images_dir = self.output_dir / "raw_images"
        self.enhanced_images_dir = self.output_dir / "enhanced_images"
        self.labels_dir = self.output_dir / "labels"
        
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.raw_images_dir, self.enhanced_images_dir, self.labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created: {dir_path}")
    
    def _get_sample_texts(self):
        """Get sample text samples for generation."""
        sample_texts = [
            "The quick brown fox jumps",
            "Handwritten text recognition",
            "Machine learning algorithm",
            "Deep neural networks",
            "Computer vision model",
            "Image processing technique",
            "Hello world example",
            "Sample training data",
            "Feature extraction method",
            "Text recognition system",
            "Natural language processing",
            "Artificial intelligence today",
            "Data science and analytics",
            "Python programming language",
            "Software development kit",
            "Research and development",
            "Quality assurance testing",
            "User experience design",
            "Mobile application development",
            "Cloud computing services",
            "Database management system",
            "Network security protocol",
            "Cybersecurity framework",
            "Information technology",
            "Digital transformation",
            "Smart automation system",
            "Internet of things devices",
            "Blockchain technology",
            "Cryptocurrency exchange",
            "Financial technology sector",
            "Healthcare digital platform",
            "Education learning system",
            "Transportation logistics",
            "Retail commerce industry",
            "Manufacturing production",
            "Energy sustainable solution",
            "Environment conservation",
            "Agriculture farming modern",
            "Construction building project",
            "Real estate property",
            "Tourism hospitality service",
            "Entertainment media industry",
            "Sports athletic competition",
            "Music audio production",
            "Photography visual art",
            "Fashion clothing design",
            "Food restaurant service",
            "Travel journey adventure",
            "Science research discovery",
            "Mathematics calculation",
            "Physics quantum mechanics",
        ]
        return sample_texts
    
    def generate_synthetic_image(self, text, image_size=(128, 128)):
        """
        Generate synthetic handwritten-style text image.
        
        Args:
            text: Text to render
            image_size: Size of output image (height, width)
            
        Returns:
            Image array
        """
        # Create white background
        img = Image.new('L', (image_size[1], image_size[0]), color=255)
        draw = ImageDraw.Draw(img)
        
        # Try to use a system font, fallback to default
        try:
            # Try different font paths for Windows
            font_paths = [
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\calibri.ttf",
                "C:\\Windows\\Fonts\\times.ttf",
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, size=16)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Add some random variations to simulate handwriting
        text_with_noise = text[:20] if len(text) > 20 else text
        
        # Draw text with slight random offset
        x = random.randint(5, 15)
        y = random.randint(40, 60)
        draw.text((x, y), text_with_noise, fill=0, font=font)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Add random noise and distortion
        if random.random() > 0.3:
            # Add Gaussian noise
            noise = np.random.normal(0, 10, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Add slight rotation
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            h, w = img_array.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_array = cv2.warpAffine(img_array, M, (w, h), 
                                      borderValue=255)
        
        return img_array
    
    def generate_dataset(self):
        """Generate complete dataset."""
        logger.info(f"Generating {self.num_samples} sample images...")
        
        sample_texts = self._get_sample_texts()
        labels = []
        
        for idx in range(self.num_samples):
            # Select random text
            text = random.choice(sample_texts)
            
            # Generate image
            img_array = self.generate_synthetic_image(text)
            
            # Save raw image
            img_filename = f"sample_{idx:04d}.png"
            img_path = self.raw_images_dir / img_filename
            cv2.imwrite(str(img_path), img_array)
            
            # Create label entry
            label = {
                "image": img_filename,
                "text": text,
                "id": idx
            }
            labels.append(label)
            
            if (idx + 1) % 20 == 0:
                logger.info(f"Generated {idx + 1}/{self.num_samples} images")
        
        # Save labels to JSON
        labels_path = self.labels_dir / "labels.json"
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
        
        logger.info(f"Dataset generation completed!")
        logger.info(f"Raw images: {self.raw_images_dir}")
        logger.info(f"Labels: {labels_path}")
        
        return labels
    
    def create_train_test_split(self, train_ratio=0.8):
        """
        Create train/test split.
        
        Args:
            train_ratio: Ratio of training data
            
        Returns:
            Train and test labels
        """
        labels_path = self.labels_dir / "labels.json"
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        # Shuffle and split
        random.shuffle(labels)
        split_idx = int(len(labels) * train_ratio)
        
        train_labels = labels[:split_idx]
        test_labels = labels[split_idx:]
        
        # Save split
        with open(self.labels_dir / "train_labels.json", 'w') as f:
            json.dump(train_labels, f, indent=2)
        
        with open(self.labels_dir / "test_labels.json", 'w') as f:
            json.dump(test_labels, f, indent=2)
        
        logger.info(f"Train samples: {len(train_labels)}")
        logger.info(f"Test samples: {len(test_labels)}")
        
        return train_labels, test_labels


def main():
    """Main execution function."""
    # Generate dataset
    generator = SampleDatasetGenerator(output_dir="dataset", num_samples=100)
    labels = generator.generate_dataset()
    train_labels, test_labels = generator.create_train_test_split()
    
    logger.info("\nDataset Summary:")
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Train samples: {len(train_labels)}")
    logger.info(f"Test samples: {len(test_labels)}")
    logger.info(f"\nDataset location: dataset/")
    logger.info(f"Images: dataset/raw_images/")
    logger.info(f"Labels: dataset/labels/")


if __name__ == "__main__":
    main()
