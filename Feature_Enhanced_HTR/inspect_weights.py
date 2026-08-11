import h5py

def print_shapes(name, node):
    if isinstance(node, h5py.Dataset):
        print(f"{name}: {node.shape}")

with h5py.File('checkpoints/best_model.h5', 'r') as f:
    f.visititems(print_shapes)
