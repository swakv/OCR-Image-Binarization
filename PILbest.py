
from PIL import Image
from PIL import ImageFilter, ImageEnhance
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('outputs/my-sharpened-image_1.jpg')
# im = Image.open('outputs/my-sharpened-image.jpg')
white = im.filter(ImageFilter.BLUR).filter(ImageFilter.MaxFilter(15))

grey = im.convert('L')
width,height = im.size
impix = im.load()
whitepix = white.load()
greypix = grey.load()

# increases the contrast in the regions with poor lighting
for y in range(height):
    for x in range(width):
        greypix[x,y] = min(255, max(255 + impix[x,y][0] - whitepix[x,y][0], 255 + impix[x,y][1] - whitepix[x,y][1], 255 + impix[x,y][2] - whitepix[x,y][2]))

grey.show()

result = pytesseract.image_to_string(grey, lang="eng")
print(result)

grey.save("best_1.png")