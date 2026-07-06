import cv2
import pytesseract
import os

for f in os.listdir('uploads'):
    path = os.path.join('uploads', f)
    if not path.endswith(('.png', '.jpg')): continue
    print(f"\n--- {f} ---")
    try:
        text = pytesseract.image_to_string(cv2.imread(path))
        print(text.strip())
    except Exception as e:
        print(e)
