import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageFilter, ImageEnhance
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def overallBrightness(im):
    white = im.filter(ImageFilter.BLUR).filter(ImageFilter.MaxFilter(15))
    grey = im.convert('L')
    width,height = im.size
    impix = im.load()
    whitepix = white.load()
    greypix = grey.load()
    for y in range(height):
        for x in range(width):
            greypix[x,y] = min(255, max(255 + impix[x,y][0] - whitepix[x,y][0], 255 + impix[x,y][1] - whitepix[x,y][1], 255 + impix[x,y][2] - whitepix[x,y][2]))
    grey.save("temp/temp2.png")

def fix_brightness(img):
    img_dot = img
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    y,x,z = img.shape
    l_blur = cv2.GaussianBlur(l, (11, 11), 5)
    maxval = []
    count_percent = 3 
    count_percent = count_percent/100
    row_percent = int(count_percent*x) 
    column_percent = int(count_percent*y) 
    for i in range(1,x-1):
        if i%row_percent == 0:
            for j in range(1, y-1):
                if j%column_percent == 0:
                    pix_cord = (i,j)
                    img_segment = l_blur[i:i+3, j:j+3]
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                    maxval.append(maxVal)

    avg_maxval = round(sum(maxval) / len(maxval))
    if avg_maxval>35 and avg_maxval<45:
        norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = np.clip(norm_img2, 0, 1)
        norm_img2 = (255*norm_img2).astype(np.uint8)
        img = norm_img2

    elif avg_maxval<35 and avg_maxval>30:
        norm_img2 = cv2.normalize(img, None, alpha=0, beta=2.4, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = np.clip(norm_img2, 0, 1)
        norm_img2 = (255*norm_img2).astype(np.uint8)
        img = norm_img2

    elif avg_maxval <30:
        norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.35, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = np.clip(norm_img2, 0, 1)
        norm_img2 = (255*norm_img2).astype(np.uint8)
        img = norm_img2

    elif avg_maxval > 70:
        norm_img2 = cv2.normalize(img, None, alpha=0, beta=0.75, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = np.clip(norm_img2, 0, 1)
        norm_img2 = (255*norm_img2).astype(np.uint8)
        img = norm_img2

    elif avg_maxval > 60:
        norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = np.clip(norm_img2, 0, 1)
        norm_img2 = (255*norm_img2).astype(np.uint8)
        img = norm_img2

    else:
        norm_img2 = cv2.normalize(img, None, alpha=0, beta=0.85, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = np.clip(norm_img2, 0, 1)
        norm_img2 = (255*norm_img2).astype(np.uint8)
        img = norm_img2

    return img

image = cv.imread('samples/sample01.png')
cv.imshow("original image", image)
cv.waitKey(0)
sharpened_image = unsharp_mask(image)
cv.imshow("sharpened image", sharpened_image)
cv.waitKey(0)
cv.imwrite("temp.png", sharpened_image)

im = Image.open('temp/temp.png')
brightness_1 = overallBrightness(im)
brightness_1 = cv.imread('temp/temp2.png')
cv2.imshow("Overall Brighness Fix", brightness_1)
cv2.waitKey(0)

w, h, c = brightness_1.shape
brightness_1[0:w//2, 0:h//2] = fix_brightness(brightness_1[0:w//2, 0:h//2])
brightness_1[w//2:w, h//2:h] = fix_brightness(brightness_1[w//2:w, h//2:h])
brightness_1[w//2:w, 0:h//2] = fix_brightness(brightness_1[w//2:w, 0:h//2])
brightness_1[0:w//2, h//2:h] = fix_brightness(brightness_1[0:w//2, h//2:h])
cv2.imshow("fix brightness", brightness_1)
img = cv2.cvtColor(brightness_1, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(img)
print(text)
cv2.waitKey(0)

img = cv2.GaussianBlur(img, (3, 3), 0)
thresh4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 11)
cv2.imshow("ADAPTIVE", thresh4)
# text = pytesseract.image_to_string(thresh4)
# print(text)
cv2.waitKey(0)

