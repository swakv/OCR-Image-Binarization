import numpy as np
import cv2
import pytesseract


image = cv2.imread('../samples/sample01.png', 0)
cv2.imshow("Original Image", image)
cv2.waitKey(0)


def adaptive_thresh(image):

    window_size = image.shape[1]/5
    delta = 11

    #output img
    output = np.zeros_like(image)    

    for col in range(image.shape[1]):
        for row in range(image.shape[0]):
            y0 = int(max(row-window_size, 0))
            y1 = int(min(row+window_size, image.shape[0]-1))
            x0 = int(max(col-window_size, 0))
            x1 = int(min(col+window_size, image.shape[1]-1))

            window_count = (y1-y0) * (x1-x0)



            temp = image[y0:y1+1, x0:x1+1]
            sums = np.sum(temp)

            t = (sums/window_count) - delta

            if image[row, col] < t:
                output[row,col] = 0
            else:
                output[row,col] = 255

    return output

image = adaptive_thresh(image)
cv2.imshow("Adaptive Thresholding", image)
text = pytesseract.image_to_string(image)
print(text)
cv2.waitKey(0)
