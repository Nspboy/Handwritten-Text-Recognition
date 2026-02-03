# Enhanced Handwritten Text Recognition (FEHR)

## Quick Start Guide

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

**Single Image Recognition:**

```bash
python main.py --mode predict --image sample.png --model checkpoints/best_model.h5
```

**Batch Processing:**

```bash
python main.py --mode predict --input-dir dataset/raw_images/ --output results.txt --model checkpoints/best_model.h5
```

**Training:**

```bash
python train.py --config config.json
```

## Key Features

✓ **Complete HTR Pipeline**: Preprocessing → CNN → BiLSTM → HRNN+Attention → CTC → NLP
✓ **Production-Ready Code**: Error handling, logging, configuration management
✓ **Modular Architecture**: Easily swappable components
✓ **Advanced Techniques**:

- Multi-head attention for feature enhancement
- Residual connections for better gradient flow
- CTC loss for alignment-free training
- Configurable text correction strategies

## Configuration

Edit `config.json` to customize:

- Model architecture
- Training parameters
- Input/output paths
- NLP settings

## Performance

For optimal results:

- Ensure high-quality handwritten text images
- Use 128×128 pixel input size (configurable)
- Train for 50-100 epochs depending on dataset
- Use appropriate batch size based on GPU memory

## Troubleshooting

| Issue          | Solution                                              |
| -------------- | ----------------------------------------------------- |
| Out of Memory  | Reduce batch_size in config.json                      |
| Poor accuracy  | Ensure good image preprocessing, increase epochs      |
| Slow inference | Use GPU, disable text correction with --no-correction |

## Next Steps

1. Prepare your dataset (images + ground truth labels)
2. Update `config.json` with dataset paths
3. Run training: `python train.py`
4. Use trained model for prediction: `python main.py --image test.png`

See **README.md** for detailed documentation.
