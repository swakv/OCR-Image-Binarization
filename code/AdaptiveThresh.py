import numpy as np
import cv2
import pytesseract


image = cv2.imread('../samples/sample02.png', 0)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

def adaptive_thresh(image):

    window_size = image.shape[1]/16
    delta = 11

    #integral img
    img = np.zeros_like(image, dtype=np.uint32)
    for col in range(image.shape[1]):
        for row in range(image.shape[0]):
            img[row,col] = image[0:row,0:col].sum()

    #output img
    output = np.zeros_like(image)    

    for col in range(image.shape[1]):
        for row in range(image.shape[0]):
            y0 = int(max(row-window_size, 0))
            y1 = int(min(row+window_size, image.shape[0]-1))
            x0 = int(max(col-window_size, 0))
            x1 = int(min(col+window_size, image.shape[1]-1))

            window_count = (y1-y0) * (x1-x0)

            sums = img[y1, x1] - img[y0, x1] - img[y1, x0] + img[y0, x0]

            if image[row, col] * window_count < sums * ((100.- delta)/100.) :
                output[row,col] = 0
            else:
                output[row,col] = 255

    return output

image = adaptive_thresh(image)
cv2.imshow("Adaptive Thresholding", image)
text = pytesseract.image_to_string(image)
print(text)
cv2.waitKey(0)
