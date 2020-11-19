import re
import skimage.io
import pytesseract
import skimage.transform
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
import cv2

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    _, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return angle
	
img_path = '../samples/test0.png'
image = cv2.imread(img_path)
im = skimage.io.imread(img_path)
imS = cv2.resize(image, (660, 540))
cv2.imshow("original image", imS)
cv2.waitKey(0)

angle_skew = deskew(image)
image = rotateImage(image, -1.0 * angle_skew)
newdata = pytesseract.image_to_osd(im, nice=1)
angle_rot = re.search('(?<=Rotate: )\d+', newdata).group(0)
rotated = ndimage.rotate(image, -float(angle_rot))

if angle_rot !=0 and angle_skew == 0:
    image = rotated

imS = cv2.resize(image, (760, 540))
cv2.imshow("Fixed Image", imS)
cv2.waitKey(0)





