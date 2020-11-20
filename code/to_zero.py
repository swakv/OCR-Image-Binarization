import numpy as np
import cv2
import pytesseract

image = cv2.imread('../samples/sample02.png', 0)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

ret2,th3 = cv2.threshold(image,120,255,cv2.THRESH_TOZERO)
cv2.imshow("TO ZERO Orginal Image", th3)
cv2.waitKey(0)

print(np.mean(image)) 
thresh = 120

for col in range(image.shape[1]):
    for row in range(image.shape[0]):
        if image[row, col] > thresh:
            image[row, col] = image[row, col]
        else:
            image[row, col] = 0



cv2.imshow("Our To Zero Image", image)
text = pytesseract.image_to_string(image)
print(text)
cv2.waitKey(0)



