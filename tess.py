# import cv2
# import numpy as np
# import pytesseract

# img = cv2.imread('outputs/best.png',0)
# cv2.imshow("IMAGE", img)
# cv2.waitKey(0)

# image=img
# img = cv2.GaussianBlur(image, (3, 3), 0)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 9)
# cv2.imshow("img", img)
# result = pytesseract.image_to_string(img, lang="eng")
# print(result)
# cv2.waitKey(0)

# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import pytesseract

# img = cv2.imread('outputs/best.png',0)
# cv2.imshow("IMAGE", img)
# cv2.waitKey(0)

# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 7)
# cv2.imshow("img", img)
# cv2.waitKey(0)

# # k = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1,1))
# # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
# # cv2.imshow("closing", closing)
# # cv2.waitKey(0)

# k1 = np.ones((1, 1), np.uint8)
# dilate = cv2.dilate(img, k1, iterations=3) 
# cv2.imshow("dilate", dilate)

# # k1 = np.ones((2, 2), np.uint8)
# # erosion = cv2.erode(dilate, k1, iterations = 1)
# # cv2.imshow("erosion", erosion)

# result = pytesseract.image_to_string(dilate, lang="eng")
# print(result)
# cv2.waitKey(0)

import cv2
import numpy as np

image = cv2.imread('outputs/best.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 275, 255, cv2.THRESH_TRUNC)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    area = cv2.contourArea(c)
    if area < 150:
        cv2.drawContours(opening, [c], -1, (0,0,0), -1)

result = 255 - opening 
cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('result', result)
cv2.waitKey()