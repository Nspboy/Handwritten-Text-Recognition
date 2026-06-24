import hashlib
import os

for root, dirs, files in os.walk('.'):
    if 'venv' in root or 'node_modules' in root or '.git' in root:
        continue
    for f in files:
        if f.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, f)
            with open(path, 'rb') as file:
                data = file.read()
                print(f, hashlib.md5(data).hexdigest(), path)
