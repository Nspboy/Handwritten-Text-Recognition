import hashlib
import os

custom_dir = 'data/custom'
if os.path.exists(custom_dir):
    for f in os.listdir(custom_dir):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            with open(os.path.join(custom_dir, f), 'rb') as file:
                data = file.read()
                print(f, hashlib.md5(data).hexdigest())
