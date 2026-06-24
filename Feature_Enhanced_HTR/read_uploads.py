import cv2
import pytesseract
import os

for f in os.listdir('uploads'):
    if f.endswith('.png') or f.endswith('.jpg'):
        img = cv2.imread(os.path.join('uploads', f))
        print('FILE:', f)
        print(pytesseract.image_to_string(img).strip())
        print('-'*20)
