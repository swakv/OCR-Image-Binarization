import numpy as np
import cv2
import pytesseract

image = cv2.imread('../samples/sample02.png', 0)
cv2.imshow("Original Image", image)
cv2.waitKey(0)


print(np.mean(image)) # 120
thresh = 120

for col in range(image.shape[1]):
    for row in range(image.shape[0]):
        if image[row, col] > thresh:
            image[row, col] = 255
        else:
            image[row, col] = 0

cv2.imshow("Our Binary Image", image)
text = pytesseract.image_to_string(image)
print(text)
cv2.waitKey(0)



