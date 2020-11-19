import cv2
import numpy as np
import pytesseract

img = cv2.imread('/home/nishka/Desktop/CV_Project/my-sharpened-image.jpg',0)
cv2.imshow("IMAGE", img)
cv2.waitKey(0)

image=img
img = cv2.GaussianBlur(image, (3, 3), 0)

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 7)
cv2.imshow("img", img)
cv2.waitKey(0)
kernel = np.ones((2,2),np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=2) 
img_dilation = cv2.dilate(img, kernel, iterations=1) 
cv2.imshow("dil", img_dilation)
result = pytesseract.image_to_string(img_dilation, lang="eng")
print(result)
cv2.waitKey(0)

cv2.destroyAllWindows()



