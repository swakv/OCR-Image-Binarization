import pytesseract
import cv2
import os
from PIL import Image

image = cv2.imread("../samples/sample02.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray)
print(text)
print("---------------------------")

