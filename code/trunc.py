import numpy as np
import cv2
import pytesseract

image = cv2.imread('../samples/sample02.png', 0)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

print(np.mean(image)) 
thresh = 120

for col in range(image.shape[1]):
    for row in range(image.shape[0]):
        if image[row, col] > thresh:
            image[row, col] = thresh
        else:
            image[row, col] = image[row, col]



cv2.imshow("Our TRUNC Image", image)
text = pytesseract.image_to_string(image)
print(text)
cv2.waitKey(0)


