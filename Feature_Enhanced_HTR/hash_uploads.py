import hashlib
import os

for f in os.listdir('uploads'):
    if f.endswith(('.png', '.jpg')):
        with open(os.path.join('uploads', f), 'rb') as file:
            data = file.read()
            print(f, hashlib.md5(data).hexdigest())
