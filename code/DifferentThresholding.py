import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

img = cv2.imread('/home/nishka/Desktop/CV_Project/sample01.png',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range (6):
    cv2.imshow(titles[i], images[i])
    print("-----------------------------------------------")
    print(titles[i])
    text = pytesseract.image_to_string(images[i])
    print(text)
    print()
    cv2.waitKey(0)
cv2.destroyAllWindows()
