# 📑 Feature Enhanced HTR - Complete Documentation Index

## Quick Navigation

### 🚀 Getting Started

- **[GETTING_STARTED.md](GETTING_STARTED.md)** ⭐ **START HERE**
  - Installation steps
  - Quick examples
  - Training tutorial
  - Prediction guide
  - Troubleshooting

### 📚 Documentation Files

#### Core Documentation

1. **[README.md](README.md)** - Comprehensive User Guide
   - Architecture overview
   - Module documentation
   - Complete API
   - Advanced features
   - Performance optimization
   - References

2. **[QUICKSTART.md](QUICKSTART.md)** - Quick Reference
   - Installation summary
   - Basic usage
   - Key features
   - Common commands
   - Performance tips

3. **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API Documentation
   - Module dependency graph
   - Full API reference for all classes
   - Data flow examples
   - Configuration schema
   - Error handling patterns

4. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation Details
   - Project overview
   - Completed implementation checklist
   - Module descriptions
   - Architecture layers
   - Key improvements
   - Extension points

#### Configuration

5. **[config.json](config.json)** - Configuration Template
   - All configurable parameters
   - Default values
   - Sub-configurations for each module

---

## 📋 Project Contents

### Core Modules

#### 1. **preprocessing/** - Image Preprocessing

- `preprocess.py` - ImagePreprocessor class
  - Grayscale conversion
  - Gaussian blur
  - Binary thresholding
  - Morphological operations
  - Batch processing

#### 2. **model/** - Deep Learning Models

- `cnn_feature_extractor.py` - CNN Features
  - 3-layer CNN (32→64→128 filters)
  - BatchNormalization
  - MaxPooling
  - Dropout regularization

- `sequence_model.py` - Sequence Modeling
  - Bidirectional LSTM
  - Multi-layer stacking
  - CNN-to-sequence integration
  - LSTM with attention

- `enhancement_hrnn.py` - Feature Enhancement
  - Multi-head attention (4 heads)
  - Residual connections
  - Layer normalization
  - Cross-modal attention

- `decoder_ctc.py` - CTC Decoding
  - CTC loss computation
  - Greedy decoding
  - Beam search decoding
  - Custom CTC layer

#### 3. **nlp/** - Text Post-Processing

- `postprocess.py`
  - TextCorrector (Simple, SymSpell, Transformer)
  - TextNormalizer
  - LanguageModel
  - Confidence scoring

### Execution Files

1. **train.py** - Training Pipeline
   - HTRTrainer class
   - Model building
   - Training loop
   - Callbacks (checkpointing, early stopping, etc.)
   - Model persistence

2. **main.py** - Inference Pipeline
   - HTRPipeline class
   - Single image prediction
   - Batch processing
   - Command-line interface
   - Results export

3. **utils.py** - Utility Functions
   - Config management
   - Data utilities
   - File utilities
   - Metrics calculation

---

## 🎯 Common Tasks

### Installation

```bash
pip install -r requirements.txt
```

→ See [GETTING_STARTED.md](GETTING_STARTED.md#installation)

### Training Model

```bash
python train.py --config config.json
```

→ See [GETTING_STARTED.md](GETTING_STARTED.md#training-your-model)

### Single Prediction

```bash
python main.py --image sample.png --model best_model.h5
```

→ See [GETTING_STARTED.md](GETTING_STARTED.md#making-predictions)

### Batch Processing

```bash
python main.py --input-dir images/ --output results.txt --model best_model.h5
```

→ See [README.md](README.md#batch-processing)

### Preprocessing Images

```python
from preprocessing.preprocess import ImagePreprocessor
preprocessor = ImagePreprocessor()
count = preprocessor.batch_preprocess("raw/", "enhanced/")
```

→ See [API_REFERENCE.md](API_REFERENCE.md#preprocessingpreprocess)

### Text Correction

```python
from nlp.postprocess import TextCorrector
corrector = TextCorrector()
corrected = corrector.correct_text("teh quick fox")
```

→ See [API_REFERENCE.md](API_REFERENCE.md#nlppostprocess)

---

## 📊 Architecture Overview

```
Input Image → Preprocessing → CNN → BiLSTM → HRNN+Attention → CTC → NLP → Output
```

**Detailed Flow**:

```
Raw Image (any size)
    ↓
[ImagePreprocessor]
    ├─ Grayscale conversion
    ├─ Gaussian blur
    ├─ Binary thresholding
    └─ Morphological ops
    ↓
Normalized Image (128×128×1)
    ↓
[CNNFeatureExtractor]
    ├─ Conv(32) + BatchNorm + MaxPool
    ├─ Conv(64) + BatchNorm + MaxPool
    └─ Conv(128) + BatchNorm + MaxPool
    ↓
Feature Maps (16×16×128)
    ↓
[BiLSTMSequenceModel]
    ├─ Reshape to sequence (256, 128)
    ├─ BiLSTM Layer 1 (128 units)
    └─ BiLSTM Layer 2 (128 units)
    ↓
Encoded Sequences (256, 256)
    ↓
[HierarchicalRNNEnhancer]
    ├─ Multi-head Attention (4 heads)
    ├─ Residual Connections
    ├─ Layer Normalization
    └─ Feed-Forward Network
    ↓
Enhanced Features (256, 256)
    ↓
[CTCDecoder]
    ├─ Dense layer to class probabilities
    └─ CTC loss computation
    ↓
Character Probabilities
    ↓
[CTC Decoding]
    ├─ Greedy decoding or
    └─ Beam search decoding
    ↓
Character Indices
    ↓
[TextCorrector]
    ├─ Spell correction
    ├─ Grammar fixes
    └─ Text normalization
    ↓
Final Recognized Text
```

---

## 🔧 Configuration Guide

### Basic Configuration

```json
{
  "batch_size": 32,
  "epochs": 100,
  "learning_rate": 0.001,
  "input_shape": [128, 128, 1],
  "num_classes": 80
}
```

### Advanced Configuration

- **CNN Parameters**: Filter sizes, activation functions
- **LSTM Parameters**: Units, layers, return_sequences
- **Attention**: Number of heads, dropout rates
- **NLP**: Correction method, language model path
- **Training**: Learning rate scheduling, callbacks

→ See [config.json](config.json) for full template

---

## 📈 Performance Tips

### Training

- Use GPU for ~10x speedup
- Monitor TensorBoard: `tensorboard --logdir=logs/`
- Increase batch size for better GPU utilization
- Adjust learning rate if loss plateaus

### Inference

- Batch process images for efficiency
- Cache the loaded model
- Use `--no-correction` for speed
- Prefer greedy decoding over beam search

### Memory

- Reduce batch size if OOM
- Decrease input image dimensions
- Use mixed precision training

→ See [README.md](README.md#performance-optimization)

---

## 🐛 Troubleshooting

### Common Issues

| Issue            | Solution                             | Reference                                                                |
| ---------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| Out of Memory    | Reduce batch size, image size        | [GETTING_STARTED.md](GETTING_STARTED.md#issue-out-of-memory-oom)         |
| Poor Accuracy    | Check preprocessing, increase epochs | [GETTING_STARTED.md](GETTING_STARTED.md#issue-poor-recognition-accuracy) |
| Slow Inference   | Disable correction, reduce size      | [GETTING_STARTED.md](GETTING_STARTED.md#issue-slow-inference)            |
| Import Errors    | Reinstall dependencies               | [GETTING_STARTED.md](GETTING_STARTED.md#issue-module-import-errors)      |
| GPU Not Detected | Install tensorflow-gpu, CUDA         | [GETTING_STARTED.md](GETTING_STARTED.md#issue-cudagpu-not-detected)      |

---

## 📦 Dependencies

### Required

- TensorFlow 2.10+
- OpenCV 4.0+
- NumPy 1.23+
- Pillow 6.0+

### Optional

- transformers 4.0+ (advanced NLP)
- torch 1.7+ (for transformers)
- symspellpy 6.0+ (spell correction)

→ See [requirements.txt](requirements.txt)

---

## 🔍 Module Overview

| Module            | Purpose          | Classes                                 | Key Methods                        |
| ----------------- | ---------------- | --------------------------------------- | ---------------------------------- |
| preprocessing     | Image processing | ImagePreprocessor                       | preprocess_image, batch_preprocess |
| model/cnn         | CNN features     | CNNFeatureExtractor                     | build_cnn, build_cnn_functional    |
| model/sequence    | BiLSTM           | BiLSTMSequenceModel                     | build_with_cnn_output              |
| model/enhancement | Attention        | HierarchicalRNNEnhancer                 | build_enhancement_model            |
| model/decoder     | CTC              | CTCDecoder                              | ctc_loss, ctc_decode               |
| nlp               | Text correction  | TextCorrector, TextNormalizer           | correct_text, normalize            |
| train             | Training         | HTRTrainer                              | build_model, train, save_model     |
| main              | Inference        | HTRPipeline                             | process_image_file, process_batch  |
| utils             | Utilities        | Config, DataUtil, FileUtil, MetricsUtil | load, save, calculate              |

---

## 🎓 Learning Path

1. **Start Here**: [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Quick Overview**: [QUICKSTART.md](QUICKSTART.md)
3. **Detailed Guide**: [README.md](README.md)
4. **API Details**: [API_REFERENCE.md](API_REFERENCE.md)
5. **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## ✅ Code Quality Standards

- ✓ PEP 8 compliant code
- ✓ Type hints on all functions
- ✓ Google-style docstrings
- ✓ Comprehensive error handling
- ✓ Detailed logging at all levels
- ✓ Input validation
- ✓ Resource cleanup
- ✓ SOLID principles
- ✓ DRY (Don't Repeat Yourself)
- ✓ Clear separation of concerns

---

## 📞 Support

**For Questions About**:

- Installation → [GETTING_STARTED.md](GETTING_STARTED.md)
- Training → [README.md](README.md#training-details)
- Prediction → [QUICKSTART.md](QUICKSTART.md)
- API Usage → [API_REFERENCE.md](API_REFERENCE.md)
- Troubleshooting → [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting)
- Configuration → [config.json](config.json)

---

## 🚀 Quick Start Command

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (if you have data)
python train.py --config config.json

# 3. Predict
python main.py --image sample.png --model checkpoints/best_model.h5

# 4. Batch process
python main.py --input-dir dataset/raw_images/ --output results.txt --model checkpoints/best_model.h5
```

---

**Version**: 1.0.0  
**Last Updated**: February 2026  
**Status**: ✅ Production Ready

For detailed information, start with [GETTING_STARTED.md](GETTING_STARTED.md)
