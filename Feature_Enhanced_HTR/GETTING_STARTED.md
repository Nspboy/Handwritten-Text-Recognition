# 🚀 Complete Getting Started Guide - Feature Enhanced HTR

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Quick Examples](#quick-examples)
5. [Training Your Model](#training-your-model)
6. [Making Predictions](#making-predictions)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Feature Enhanced Handwritten Text Recognition (FEHR)** is a production-ready system that recognizes handwritten text using:

- **🖼️ Preprocessing**: Image enhancement with morphological operations
- **🧠 CNN**: Deep convolutional features (128 filters, 3 layers)
- **📊 BiLSTM**: Temporal sequence modeling (bidirectional, 2 layers)
- **⚡ HRNN+Attention**: Hierarchical RNN with multi-head attention (4 heads)
- **🎯 CTC**: Connectionist Temporal Classification (alignment-free)
- **📝 NLP**: Text correction and normalization

**Key Advantages**:

- ✅ Alignment-free training with CTC loss
- ✅ Attention mechanisms for feature enhancement
- ✅ Post-processing text correction
- ✅ Production-ready error handling
- ✅ Fully configurable via JSON
- ✅ GPU acceleration support

---

## Installation

### Step 1: Prerequisites

```bash
# Ensure Python 3.8+ installed
python --version

# Ensure pip is up to date
python -m pip install --upgrade pip
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
cd Feature_Enhanced_HTR
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## Project Structure

```
Feature_Enhanced_HTR/
│
├── 📁 dataset/
│   ├── raw_images/          ← Place your input images here
│   ├── enhanced_images/     ← Preprocessed images (auto-generated)
│   └── labels/              ← Ground truth labels (training)
│
├── 📁 preprocessing/
│   ├── __init__.py
│   └── preprocess.py        ← Image preprocessing module
│
├── 📁 model/
│   ├── __init__.py
│   ├── cnn_feature_extractor.py    ← CNN module
│   ├── sequence_model.py           ← BiLSTM module
│   ├── enhancement_hrnn.py         ← Attention module
│   └── decoder_ctc.py              ← CTC loss/decoding
│
├── 📁 nlp/
│   ├── __init__.py
│   └── postprocess.py       ← Text correction module
│
├── 🔧 Main Files
│   ├── train.py             ← Training script
│   ├── main.py              ← Prediction script
│   ├── utils.py             ← Utility functions
│   ├── config.json          ← Configuration file
│   └── requirements.txt      ← Python dependencies
│
└── 📚 Documentation
    ├── README.md            ← Full documentation
    ├── QUICKSTART.md        ← Quick reference
    ├── API_REFERENCE.md     ← API documentation
    └── GETTING_STARTED.md   ← This file
```

---

## Quick Examples

### 1️⃣ Preprocess Images

```python
from preprocessing.preprocess import ImagePreprocessor

# Initialize preprocessor
preprocessor = ImagePreprocessor(
    blur_kernel=(5, 5),
    morphology_enabled=True
)

# Option A: Preprocess single image
processed = preprocessor.preprocess_image("dataset/raw_images/sample.png")
print(f"Processed shape: {processed.shape}")

# Option B: Batch process directory
count = preprocessor.batch_preprocess(
    input_dir="dataset/raw_images/",
    output_dir="dataset/enhanced_images/"
)
print(f"Processed {count} images")
```

### 2️⃣ Build Model Architecture

```python
from train import HTRTrainer

# Initialize trainer
trainer = HTRTrainer(config_path="config.json")

# Build model
model = trainer.build_model()

# View architecture
trainer.model_summary()
```

### 3️⃣ Train the Model

```python
import numpy as np
from train import HTRTrainer

trainer = HTRTrainer(config_path="config.json")
model = trainer.build_model()

# Prepare your data
# x_train: (num_samples, 128, 128, 1) - normalized to [0, 1]
# y_train: (num_samples, max_label_length) - character indices

x_train = np.random.randn(100, 128, 128, 1)  # Dummy data
y_train = np.random.randint(0, 80, (100, 32))  # Dummy labels

# Train
history = trainer.train(x_train, y_train)

# Save model
trainer.save_model("checkpoints/best_model.h5")
```

### 4️⃣ Make Predictions

```python
from main import HTRPipeline

# Initialize with trained model
pipeline = HTRPipeline(model_path="checkpoints/best_model.h5")

# Recognize text from image
text = pipeline.process_image_file(
    "dataset/raw_images/sample.png",
    apply_correction=True
)

print(f"Recognized text: {text}")
```

### 5️⃣ Batch Process Images

```python
from main import HTRPipeline

pipeline = HTRPipeline(model_path="checkpoints/best_model.h5")

# Process all images in directory
results = pipeline.process_batch(
    input_dir="dataset/raw_images/",
    output_file="results.txt",
    apply_correction=True
)

# Display results
for item in results:
    print(f"{item['image']}: {item['text']}")
```

### 6️⃣ Text Correction

```python
from nlp.postprocess import TextCorrector, TextNormalizer

# Initialize corrector
corrector = TextCorrector(use_transformers=False)
normalizer = TextNormalizer()

# Correct text
text = "teh quick brwon fox"
corrected = corrector.correct_text(text, method='simple')
normalized = normalizer.normalize(corrected)

print(f"Original:    {text}")
print(f"Corrected:   {corrected}")
print(f"Normalized:  {normalized}")
```

---

## Training Your Model

### Step 1: Prepare Dataset

Create a dataset structure:

```
dataset/
├── raw_images/
│   ├── word_1.png
│   ├── word_2.png
│   └── ...
└── labels/
    ├── word_1.txt  (containing the ground truth text)
    ├── word_2.txt
    └── ...
```

### Step 2: Update Configuration

Edit `config.json`:

```json
{
  "dataset_path": "dataset/",
  "batch_size": 32,
  "epochs": 100,
  "learning_rate": 0.001,
  "input_shape": [128, 128, 1],
  "num_classes": 80,
  "validation_split": 0.1,
  "early_stopping_patience": 10
}
```

### Step 3: Load and Preprocess Data

```python
import numpy as np
from preprocessing.preprocess import ImagePreprocessor
from pathlib import Path

preprocessor = ImagePreprocessor()

# Preprocess all images
preprocessor.batch_preprocess("dataset/raw_images/", "dataset/enhanced_images/")

# Load images and labels
def load_dataset(img_dir, label_dir, max_images=None):
    x_data = []
    y_data = []

    for i, img_file in enumerate(Path(img_dir).glob("*.png")):
        if max_images and i >= max_images:
            break

        # Load image
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        x_data.append(img)

        # Load label
        label_file = Path(label_dir) / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file) as f:
                text = f.read().strip()
                # Convert text to character indices (1-79, 0 is blank)
                char_indices = [ord(c) - 32 for c in text[:32]]
                # Pad to length 32
                char_indices += [0] * (32 - len(char_indices))
                y_data.append(char_indices[:32])

    return np.array(x_data)[..., np.newaxis], np.array(y_data)

# Load data
x_train, y_train = load_dataset("dataset/enhanced_images/", "dataset/labels/", max_images=None)
print(f"Dataset shape: {x_train.shape}, Labels shape: {y_train.shape}")
```

### Step 4: Train

```python
from train import HTRTrainer

trainer = HTRTrainer(config_path="config.json")
model = trainer.build_model()

history = trainer.train(x_train, y_train)

trainer.save_model("checkpoints/best_model.h5")

# View training curves
import matplotlib.pyplot as plt
plt.plot(history['loss'], label='Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.show()
```

---

## Making Predictions

### Option 1: Single Image (Command Line)

```bash
python main.py --image dataset/raw_images/sample.png --model checkpoints/best_model.h5
```

### Option 2: Single Image (Python)

```python
from main import HTRPipeline

pipeline = HTRPipeline(model_path="checkpoints/best_model.h5")
text = pipeline.process_image_file("sample.png")
print(f"Result: {text}")
```

### Option 3: Batch Processing (Command Line)

```bash
python main.py --input-dir dataset/raw_images/ --output results.txt --model checkpoints/best_model.h5
```

### Option 4: Batch Processing (Python)

```python
from main import HTRPipeline

pipeline = HTRPipeline(model_path="checkpoints/best_model.h5")
results = pipeline.process_batch("dataset/raw_images/", output_file="results.txt")

for item in results:
    print(f"{item['image']}: {item['text']}")
```

### Option 5: Disable Text Correction

For faster inference (no NLP correction):

```bash
python main.py --image sample.png --model model.h5 --no-correction
```

```python
text = pipeline.process_image_file("sample.png", apply_correction=False)
```

---

## Advanced Configuration

### Custom Model Parameters

Edit `config.json`:

```json
{
  "lstm_units": 256, // Increase LSTM units (more capacity)
  "num_lstm_layers": 3, // More LSTM layers (deeper)
  "enhancement": {
    "num_heads": 8, // More attention heads
    "num_blocks": 3 // More enhancement blocks
  },
  "batch_size": 64, // Larger batch size
  "learning_rate": 0.0005 // Lower learning rate
}
```

### Custom Text Correction

```python
from nlp.postprocess import TextCorrector

# Use SymSpell correction
corrector = TextCorrector(use_transformers=False)
text = corrector.correct_text("teh quick fox", method='symspell')

# Use transformer correction (requires transformers library)
corrector = TextCorrector(use_transformers=True)
text = corrector.correct_text("teh quick fox", method='transformer')
```

### Custom Preprocessing

```python
from preprocessing.preprocess import ImagePreprocessor

class CustomPreprocessor(ImagePreprocessor):
    def custom_processing(self, img):
        # Add your custom processing
        return img

preprocessor = CustomPreprocessor()
processed = preprocessor.preprocess_image("image.png")
```

### Evaluation Metrics

```python
from utils import MetricsUtil

predictions = ["hello world", "handwritten text"]
references = ["hello world", "handwritten text"]

cer = MetricsUtil.character_error_rate(predictions, references)
wer = MetricsUtil.word_error_rate(predictions, references)

print(f"Character Error Rate: {cer:.2%}")
print(f"Word Error Rate: {wer:.2%}")
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**:

```json
{
  "batch_size": 16, // Reduce batch size
  "input_shape": [96, 96, 1] // Reduce image size
}
```

### Issue: Poor Recognition Accuracy

**Solutions**:

1. Check image quality and preprocessing
2. Increase training epochs
3. Add data augmentation
4. Verify label correctness
5. Adjust learning rate

```json
{
  "epochs": 200, // More training
  "learning_rate": 0.001, // Tune learning rate
  "early_stopping_patience": 20 // More patience
}
```

### Issue: Slow Inference

**Solutions**:

```bash
# Disable text correction
python main.py --image sample.png --no-correction

# Use greedy decoding instead of beam search (automatic)
```

### Issue: Module Import Errors

```bash
# Reinstall dependencies
pip install --upgrade tensorflow opencv-python numpy

# Verify installation
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Issue: CUDA/GPU Not Detected

```python
import tensorflow as tf

# Check available devices
print(tf.config.list_physical_devices('GPU'))

# If empty, TensorFlow will use CPU automatically
# For GPU support, install tensorflow-gpu and CUDA toolkit
```

---

## Next Steps

1. **Prepare Your Data**: Organize images and labels
2. **Adjust Configuration**: Tune hyperparameters in `config.json`
3. **Train Model**: Run training script
4. **Evaluate Results**: Check metrics and visualizations
5. **Deploy**: Use trained model for inference

---

## Resources

- **Main Documentation**: See `README.md`
- **API Reference**: See `API_REFERENCE.md`
- **Quick Reference**: See `QUICKSTART.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`

---

## Support & Help

**Common Commands**:

```bash
# Show all options
python main.py --help
python train.py --help

# List all images in directory
python -c "from utils import FileUtil; print(FileUtil.list_images('dataset/raw_images/'))"

# Check configuration
python -c "from utils import Config; import json; print(json.dumps(Config.load('config.json'), indent=2))"
```

---

**Happy Text Recognition! 🎉**

For issues or questions, refer to the comprehensive documentation files or modify the code to suit your needs.

**Version**: 1.0.0  
**Last Updated**: February 2026
