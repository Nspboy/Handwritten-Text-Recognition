
import numpy as np
import tensorflow as tf
from train import HTRTrainer
from preprocessing.preprocess import ImagePreprocessor
import matplotlib.pyplot as plt

def diagnostic_check(image_path, model_path):
    # 1. Load Preprocessor and Image
    preprocessor = ImagePreprocessor()
    img = preprocessor.preprocess_image(image_path)
    if img is None: return
    
    # Resize to match model input
    img_input = tf.image.resize(np.expand_dims(img, axis=-1), [128, 128])
    img_input = np.expand_dims(img_input, axis=0) / 255.0
    
    # 2. Load Trainer and Model
    trainer = HTRTrainer()
    trainer.load_model(model_path)
    
    # 3. Get Raw Prediction (Before CTC Decoding)
    raw_pred = trainer.model.predict(img_input)
    print(f"Prediction Shape: {raw_pred.shape}")
    
    # 4. Check Softmax Distribution
    # Use softmax to see probabilities
    probs = tf.nn.softmax(raw_pred[0]).numpy()
    max_probs = np.max(probs, axis=-1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    
    plt.subplot(1, 2, 2)
    plt.plot(max_probs)
    plt.title("Confidence over Time-steps")
    plt.ylabel("Probability")
    plt.xlabel("Time-step")
    plt.show()
    
    print("Top Predicted Indices per time-step:")
    print(np.argmax(probs, axis=-1))

if __name__ == "__main__":
    diagnostic_check("dataset/raw_images/sample_0000.png", "checkpoints/best_model.h5")
