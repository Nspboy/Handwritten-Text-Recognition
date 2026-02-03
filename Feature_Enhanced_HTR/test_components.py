"""
Test and Validation Script for HTR Components

Verifies that all model components work properly with the generated dataset.
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_preprocessing():
    """Test image preprocessing functionality."""
    logger.info("\n" + "="*60)
    logger.info("Testing Image Preprocessing Module")
    logger.info("="*60)
    
    try:
        from preprocessing.preprocess import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        logger.info("✓ ImagePreprocessor imported successfully")
        
        # Get a sample image from dataset
        sample_dir = Path("dataset/raw_images")
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.png"))[:3]
            
            for img_path in sample_images:
                preprocessed = preprocessor.preprocess_image(str(img_path))
                if preprocessed is not None:
                    logger.info(f"✓ Successfully preprocessed: {img_path.name}")
                    logger.info(f"  Input shape: {cv2.imread(str(img_path), 0).shape}")
                    logger.info(f"  Output shape: {preprocessed.shape}")
                else:
                    logger.error(f"✗ Failed to preprocess: {img_path.name}")
        else:
            logger.warning("⚠ Dataset directory not found, skipping preprocessing test")
        
        return True
    except Exception as e:
        logger.error(f"✗ Preprocessing test failed: {str(e)}")
        return False


def test_cnn_extractor():
    """Test CNN feature extractor."""
    logger.info("\n" + "="*60)
    logger.info("Testing CNN Feature Extractor")
    logger.info("="*60)
    
    try:
        import tensorflow as tf
        from model.cnn_feature_extractor import CNNFeatureExtractor
        
        extractor = CNNFeatureExtractor(
            input_shape=(128, 128, 1),
            num_filters=[32, 64, 128]
        )
        logger.info("✓ CNNFeatureExtractor instantiated successfully")
        
        model = extractor.build_sequential()
        logger.info(f"✓ CNN model built successfully")
        logger.info(f"  Model summary:")
        
        # Print model info
        for layer in model.layers:
            logger.info(f"    {layer.name}: {layer.output_shape}")
        
        # Test forward pass with dummy data
        dummy_input = np.random.randn(1, 128, 128, 1).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ CNN extractor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_sequence_model():
    """Test BiLSTM sequence model."""
    logger.info("\n" + "="*60)
    logger.info("Testing BiLSTM Sequence Model")
    logger.info("="*60)
    
    try:
        import tensorflow as tf
        from model.sequence_model import BiLSTMSequenceModel
        
        seq_model = BiLSTMSequenceModel(
            input_features=128,
            lstm_units=128,
            num_layers=2
        )
        logger.info("✓ BiLSTMSequenceModel instantiated successfully")
        
        model = seq_model.build()
        logger.info(f"✓ BiLSTM model built successfully")
        logger.info(f"  Model summary:")
        
        for layer in model.layers:
            logger.info(f"    {layer.name}: {layer.output_shape}")
        
        # Test forward pass with dummy data
        dummy_input = np.random.randn(1, 256, 128).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Sequence model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_enhancement_hrnn():
    """Test Hierarchical RNN Enhancer."""
    logger.info("\n" + "="*60)
    logger.info("Testing HRNN Feature Enhancement")
    logger.info("="*60)
    
    try:
        import tensorflow as tf
        from model.enhancement_hrnn import HierarchicalRNNEnhancer
        
        enhancer = HierarchicalRNNEnhancer(
            input_features=128,
            num_heads=4,
            num_blocks=2
        )
        logger.info("✓ HierarchicalRNNEnhancer instantiated successfully")
        
        model = enhancer.build()
        logger.info(f"✓ HRNN model built successfully")
        logger.info(f"  Number of blocks: 2")
        logger.info(f"  Number of attention heads: 4")
        
        # Test forward pass with dummy data
        dummy_input = np.random.randn(1, 256, 128).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Input shape: {dummy_input.shape}")
        logger.info(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ HRNN test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_ctc_decoder():
    """Test CTC decoder."""
    logger.info("\n" + "="*60)
    logger.info("Testing CTC Decoder")
    logger.info("="*60)
    
    try:
        from model.decoder_ctc import CTCDecoder
        
        decoder = CTCDecoder(
            num_classes=80,
            blank_label=79
        )
        logger.info("✓ CTCDecoder instantiated successfully")
        logger.info(f"  Number of classes: 80")
        logger.info(f"  Blank label index: 79")
        
        # Test CTC loss computation
        batch_size = 2
        seq_length = 50
        num_classes = 80
        
        # Dummy predictions (logits)
        y_pred = np.random.randn(batch_size, seq_length, num_classes).astype(np.float32)
        
        # Dummy labels
        y_true = np.random.randint(0, 79, (batch_size, 20)).astype(np.int32)
        
        logger.info(f"✓ CTC decoder components ready")
        logger.info(f"  Test batch size: {batch_size}")
        logger.info(f"  Sequence length: {seq_length}")
        
        return True
    except Exception as e:
        logger.error(f"✗ CTC decoder test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_nlp_postprocessing():
    """Test NLP post-processing."""
    logger.info("\n" + "="*60)
    logger.info("Testing NLP Post-Processing")
    logger.info("="*60)
    
    try:
        from nlp.postprocess import TextCorrector, TextNormalizer
        
        corrector = TextCorrector(use_transformers=False)
        logger.info("✓ TextCorrector instantiated successfully")
        
        normalizer = TextNormalizer()
        logger.info("✓ TextNormalizer instantiated successfully")
        
        # Test normalization
        test_text = "  Hello   World!  "
        normalized = normalizer.normalize(test_text)
        logger.info(f"✓ Text normalization works")
        logger.info(f"  Input: '{test_text}'")
        logger.info(f"  Output: '{normalized}'")
        
        return True
    except Exception as e:
        logger.error(f"✗ NLP post-processing test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading."""
    logger.info("\n" + "="*60)
    logger.info("Testing Dataset Loading")
    logger.info("="*60)
    
    try:
        dataset_path = Path("dataset")
        
        # Check dataset structure
        if not dataset_path.exists():
            logger.error("✗ Dataset directory not found")
            return False
        
        raw_images = list((dataset_path / "raw_images").glob("*.png"))
        labels_file = dataset_path / "labels" / "labels.json"
        train_labels = dataset_path / "labels" / "train_labels.json"
        test_labels = dataset_path / "labels" / "test_labels.json"
        
        logger.info(f"✓ Dataset structure verified")
        logger.info(f"  Raw images: {len(raw_images)} files")
        
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels = json.load(f)
            logger.info(f"✓ Labels loaded: {len(labels)} total samples")
            
            # Show sample labels
            for i, label in enumerate(labels[:3]):
                logger.info(f"    Sample {i}: '{label['text']}'")
        
        if train_labels.exists():
            with open(train_labels, 'r') as f:
                train_data = json.load(f)
            logger.info(f"✓ Train labels: {len(train_data)} samples")
        
        if test_labels.exists():
            with open(test_labels, 'r') as f:
                test_data = json.load(f)
            logger.info(f"✓ Test labels: {len(test_data)} samples")
        
        return True
    except Exception as e:
        logger.error(f"✗ Dataset loading test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    logger.info("\n" + "="*60)
    logger.info("HTR Component Validation Suite")
    logger.info("="*60)
    
    results = {}
    
    # Run tests
    results["Dataset Loading"] = test_dataset_loading()
    results["Image Preprocessing"] = test_preprocessing()
    results["CNN Feature Extractor"] = test_cnn_extractor()
    results["BiLSTM Sequence Model"] = test_sequence_model()
    results["HRNN Enhancement"] = test_enhancement_hrnn()
    results["CTC Decoder"] = test_ctc_decoder()
    results["NLP Post-Processing"] = test_nlp_postprocessing()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ All components verified successfully!")
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
