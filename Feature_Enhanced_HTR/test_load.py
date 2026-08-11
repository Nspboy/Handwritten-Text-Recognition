import os
import tensorflow as tf
from engine.trainer import HTRTrainer

trainer = HTRTrainer(config_path='config.json')
model = trainer.build_model()
import h5py
f = h5py.File('checkpoints/best_model.h5', 'r')

for layer in model.layers:
    if layer.weights:
        print(f"Trying to load weights for {layer.name}")
        try:
            # Just try setting dummy weights to see if it's the layer itself?
            pass
        except Exception as e:
            pass

print("Testing load_weights layer by layer using keras load_weights functionality")
for layer in model.layers:
    if not layer.weights:
        continue
    try:
        # In keras we can't easily load_weights on a single layer from a whole model h5,
        # but we can do it by name from the file if we load it manually.
        weight_names = [w.name for w in layer.weights]
        print(f"Layer {layer.name} expects: {[w.shape for w in layer.weights]}")
    except Exception as e:
        print(e)
