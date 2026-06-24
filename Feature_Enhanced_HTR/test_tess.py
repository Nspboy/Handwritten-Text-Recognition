import cv2
import pytesseract

img = cv2.imread('uploads/image1.png')
print("Text:", pytesseract.image_to_string(img).strip())
