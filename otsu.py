import pytesseract
import cv2
import os
from PIL import Image

image = cv2.imread("/home/nishka/Desktop/CV_Project/sample01.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1] #cv2.THRESH_BINARY | 

# gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
# show the output images
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imshow("Output", gray)
cv2.waitKey(0)
cv2.imwrite("/home/nishka/Desktop/CV_Project/OTSU.png",gray )