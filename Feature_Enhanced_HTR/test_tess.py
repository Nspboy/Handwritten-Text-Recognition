import os
import cv2
import pytesseract

img = cv2.imread('tessdata/kan.traineddata') # just a dummy image to trigger tesseract
if img is None:
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)

local_tessdata = os.path.abspath('tessdata')
tess_config = f'--tessdata-dir {local_tessdata} --psm 7'
print("TESS CONFIG:", tess_config)
try:
    res = pytesseract.image_to_string(img, lang='kan', config=tess_config)
    print("RES:", res)
except Exception as e:
    print("EXCEPTION:", e)
