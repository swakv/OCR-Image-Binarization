import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageFilter, ImageEnhance
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import re
import skimage.io
import skimage.transform
from scipy import ndimage



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

def adaptive_thresh(image):

    window_size = image.shape[1]/16
    delta = 7

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

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    _, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return angle


img_path = 'samples/sample02.png'
image = cv2.imread(img_path)
im = skimage.io.imread(img_path)
cv.imshow("original image", image)
cv.waitKey(0)


angle_skew = deskew(image)
image = rotateImage(image, -1.0 * angle_skew)
try: 
    newdata = pytesseract.image_to_osd(im, nice=1)
    angle_rot = re.search('(?<=Rotate: )\d+', newdata).group(0)
except: 
    angle_rot = 0
rotated = ndimage.rotate(image, -float(angle_rot))

if angle_rot !=0 and angle_skew == 0:
    image = rotated

cv2.imshow("Fixed Rotation Image", image)
cv2.waitKey(0)

sharpened_image = unsharp_mask(image)
cv.imshow("sharpened image", sharpened_image)
cv.waitKey(0)
cv.imwrite("temp/temp.png", sharpened_image)

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
cv2.imshow("Fix brightness", brightness_1)
img = cv2.cvtColor(brightness_1, cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

image = adaptive_thresh(img)
cv2.imshow("Adaptive Threshold", image)
text = pytesseract.image_to_string(image)
print()
print(text)
cv2.waitKey(0)
