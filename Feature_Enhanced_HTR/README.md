# Feature Enhanced Handwritten Text Recognition (FEHR)

## Overview

A comprehensive, production-ready handwritten text recognition system featuring:

- **CNN Feature Extraction**: Multi-layer convolutional neural network
- **BiLSTM Sequence Modeling**: Bidirectional LSTM for temporal sequence encoding
- **HRNN with Attention**: Hierarchical RNN with multi-head attention for feature enhancement
- **CTC Decoding**: Connectionist Temporal Classification for alignment-free recognition
- **NLP Post-Processing**: Text correction and normalization

## Architecture

```
Input Image
    ↓
[Image Preprocessing]
    ↓
[CNN Feature Extraction] (32→64→128 filters)
    ↓
[Reshape for Sequence]
    ↓
[BiLSTM Layers] (Bidirectional LSTM × 2)
    ↓
[Feature Enhancement - HRNN + Attention]
    ↓
[CTC Output Layer]
    ↓
[CTC Decoding]
    ↓
[NLP Post-Processing]
    ↓
Recognized Text
```

## Project Structure

```
Feature_Enhanced_HTR/
├── dataset/
│   ├── raw_images/          # Input handwritten text images
│   ├── enhanced_images/     # Preprocessed images
│   └── labels/              # Character labels
│
├── preprocessing/
│   └── preprocess.py        # Image preprocessing module
│
├── model/
│   ├── cnn_feature_extractor.py    # CNN feature extraction
│   ├── sequence_model.py           # BiLSTM sequence modeling
│   ├── enhancement_hrnn.py         # HRNN with attention
│   └── decoder_ctc.py              # CTC loss and decoding
│
├── nlp/
│   └── postprocess.py       # Text correction and normalization
│
├── train.py                 # Training pipeline
├── main.py                  # Main execution and prediction
├── config.json              # Configuration settings
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone or download the repository
cd Feature_Enhanced_HTR

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.json` to customize:

- Input/output directories
- Model architecture parameters (LSTM units, layers, etc.)
- Training hyperparameters (batch size, learning rate, epochs)
- NLP correction settings

## Usage

### Training

```bash
# Train from scratch
python train.py --config config.json

# The trainer will:
# 1. Build the model architecture
# 2. Load your dataset
# 3. Train with checkpoints and early stopping
# 4. Save the best model
```

### Prediction (Single Image)

```bash
# Recognize text from a single image
python main.py --mode predict --image path/to/image.png --model checkpoints/best_model.h5

# Without text correction:
python main.py --mode predict --image path/to/image.png --model checkpoints/best_model.h5 --no-correction
```

### Batch Processing

```bash
# Process all images in a directory
python main.py --mode predict --input-dir dataset/raw_images/ --output results.txt --model checkpoints/best_model.h5
```

## Module Documentation

### 1. Image Preprocessing (`preprocessing/preprocess.py`)

**Features:**

- Grayscale conversion
- Gaussian blur for noise reduction
- Otsu's binary thresholding
- Morphological operations (optional)

**Usage:**

```python
from preprocessing.preprocess import ImagePreprocessor

preprocessor = ImagePreprocessor()

# Single image
processed = preprocessor.preprocess_image("image.png")

# Batch processing
count = preprocessor.batch_preprocess("raw_images/", "enhanced_images/")
```

### 2. CNN Feature Extractor (`model/cnn_feature_extractor.py`)

**Architecture:**

- Conv2D (32 filters) → BatchNorm → MaxPool → Dropout
- Conv2D (64 filters) → BatchNorm → MaxPool → Dropout
- Conv2D (128 filters) → BatchNorm → MaxPool → Dropout

**Usage:**

```python
from model.cnn_feature_extractor import CNNFeatureExtractor

extractor = CNNFeatureExtractor(input_shape=(128, 128, 1))
model = extractor.build_cnn()
extractor.summary()
```

### 3. BiLSTM Sequence Model (`model/sequence_model.py`)

**Features:**

- Bidirectional LSTM layers
- Configurable number of layers
- Dropout for regularization
- Optional attention mechanism

**Usage:**

```python
from model.sequence_model import BiLSTMSequenceModel

seq_model = BiLSTMSequenceModel(lstm_units=128, num_layers=2)
model = seq_model.build_with_cnn_output(cnn_feature_shape=(16, 16, 128))
```

### 4. Feature Enhancement (`model/enhancement_hrnn.py`)

**Features:**

- Multi-head attention mechanism
- Residual connections
- Layer normalization
- Hierarchical architecture

**Usage:**

```python
from model.enhancement_hrnn import HierarchicalRNNEnhancer

enhancer = HierarchicalRNNEnhancer(feature_dim=256, num_heads=4)
model = enhancer.build_enhancement_model(input_shape=(32, 256), num_blocks=2)
```

### 5. CTC Decoder (`model/decoder_ctc.py`)

**Features:**

- CTC loss computation
- Greedy decoding
- Beam search decoding
- Custom CTC loss layer

**Usage:**

```python
from model.decoder_ctc import CTCDecoder

decoder = CTCDecoder(num_classes=80)
loss = CTCDecoder.ctc_loss(y_true, y_pred)
predictions, probabilities = CTCDecoder.ctc_decode(y_pred)
```

### 6. NLP Post-Processing (`nlp/postprocess.py`)

**Features:**

- Simple text cleaning
- SymSpell-based spell correction
- Transformer-based correction (optional)
- Text normalization
- Confidence scoring

**Usage:**

```python
from nlp.postprocess import TextCorrector, TextNormalizer

corrector = TextCorrector(use_transformers=False)
corrected = corrector.correct_text("the quick brwon fox")

normalizer = TextNormalizer()
normalized = normalizer.normalize("   Hello   WORLD  ")
```

## Training Details

### Loss Function

CTC (Connectionist Temporal Classification) loss - enables training without explicit character-level alignment.

### Optimizer

Adam with learning rate scheduling:

- Initial LR: 0.001
- Reduction: 50% when validation loss plateaus
- Minimum LR: 0.00001

### Callbacks

- **ModelCheckpoint**: Saves best model based on validation loss
- **EarlyStopping**: Prevents overfitting (patience=10)
- **ReduceLROnPlateau**: Adaptive learning rate reduction
- **TensorBoard**: Training visualization

## Performance Optimization

### For Training:

1. Use GPU if available (set device in config)
2. Adjust batch size based on available memory
3. Use data augmentation for better generalization
4. Monitor TensorBoard logs in `logs/` directory

### For Inference:

1. Use pre-trained models to avoid training time
2. Batch process images when possible
3. Disable text correction if speed is critical
4. Use greedy decoding instead of beam search

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size in config.json
- Reduce image input size
- Use GPU with sufficient VRAM

### Poor Recognition Accuracy

- Ensure images are properly preprocessed
- Check dataset quality and size
- Increase training epochs
- Use data augmentation
- Adjust learning rate

### Slow Inference

- Use smaller model architecture
- Disable NLP correction (`--no-correction`)
- Use GPU for acceleration
- Reduce image resolution

## Advanced Features

### Custom Character Sets

Edit the character map in `model/decoder_ctc.py` to support different languages or special characters.

### Transfer Learning

Load pre-trained weights and fine-tune:

```python
trainer = HTRTrainer()
trainer.load_model("pretrained_model.h5")
# Fine-tune with new data
```

### Model Deployment

Export to TensorFlow Lite for mobile:

```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

## Dependencies

### Core

- **TensorFlow 2.10+**: Deep learning framework
- **OpenCV 4.0+**: Image processing
- **NumPy 1.23+**: Numerical computing

### Optional

- **Transformers**: For advanced NLP correction
- **SymSpellPy**: For spell correction
- **Matplotlib**: For visualization

## References

- [CTC: Connectionist Temporal Classification](https://arxiv.org/abs/1311.5039)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition](https://arxiv.org/abs/1507.05717)

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions, bug reports, and feature requests are welcome!

## Support

For issues or questions, please refer to the module documentation or create an issue in the repository.

---

**Last Updated**: February 2026
**Version**: 1.0.0
