import numpy as np
import cv2
import pytesseract
import math
from PIL import Image

image = cv2.imread('../samples/sample02.png', 0)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

def GaussianFilter(sigma):
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
    
    return gaussian_filter, filter_size

def adaptive_thresh(image):

    # sigma = 5 for sample02
    sigma = 9 # for sample01

    gauss_filter, window_size = GaussianFilter(sigma)

    output = np.zeros_like(image)    

    for col in range(image.shape[1]-window_size):
        for row in range(image.shape[0]-window_size):
            y0 = int(max(row-window_size, row))
            y1 = int(min(row+window_size, image.shape[0]))
            x0 = int(max(col-window_size, col))
            x1 = int(min(col+window_size, image.shape[1]))

            temp = image[y0:y1, x0:x1]

            t = np.multiply(temp, gauss_filter)
            t = np.sum(t)

            if image[row, col] < t:
                output[row,col] = 0
            else:
                output[row,col] = 255
                 
    output[:,image.shape[1]-window_size:] = 255
    output[image.shape[0]-window_size:, :] = 255


    return output

image = adaptive_thresh(image)
cv2.imshow("Adaptive Thresholding", image)
text = pytesseract.image_to_string(image)
print(text)
cv2.waitKey(0)
