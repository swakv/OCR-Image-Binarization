import cv2
import numpy as np
import pytesseract

img = cv2.imread('outputs/my-sharpened-image.jpg',0)
cv2.imshow("IMAGE", img)
cv2.waitKey(0)

image=img
img = cv2.GaussianBlur(image, (3, 3), 0)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 9)
cv2.imshow("img", img)
result = pytesseract.image_to_data(img, lang="eng")
print(result)
cv2.waitKey(0)

cv2.destroyAllWindows()
