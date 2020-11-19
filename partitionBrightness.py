import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

def fix_brightness(img):
    img_dot = img
    print("INSIDE FIX BRIGHTNESS-------------")
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    y,x,z = img.shape
    l_blur = cv2.GaussianBlur(l, (11, 11), 5)
    maxval = []
    count_percent = 3 #percent of total image
    count_percent = count_percent/100
    row_percent = int(count_percent*x) #1% of total pixels widthwise
    column_percent = int(count_percent*y) #1% of total pizel height wise
    for i in range(1,x-1):
        if i%row_percent == 0:
            for j in range(1, y-1):
                if j%column_percent == 0:
                    pix_cord = (i,j)
                    img_segment = l_blur[i:i+3, j:j+3]
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                    maxval.append(maxVal)

    avg_maxval = round(sum(maxval) / len(maxval))
    print("this is    ", avg_maxval)
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


img = cv2.imread('/home/nishka/Desktop/CV_Project/sample01.png')
cv2.imshow("BEFORE fix brightness", img)
cv2.waitKey(0)
w, h, c = img.shape


# img = fix_brightness(img)

# img[:, 0:h//2] = fix_brightness(img[:, 0:h//2])
# img[:, h//2:h] = fix_brightness(img[:, h//2:h])

img[0:w//2, 0:h//2] = fix_brightness(img[0:w//2, 0:h//2])
img[w//2:w, h//2:h] = fix_brightness(img[w//2:w, h//2:h])
img[w//2:w, 0:h//2] = fix_brightness(img[w//2:w, 0:h//2])
img[0:w//2, h//2:h] = fix_brightness(img[0:w//2, h//2:h])


cv2.imshow("fix brightness", img)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ret,thresh1 = cv2.threshold(img,90,255,cv2.THRESH_BINARY)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,90,255,cv2.THRESH_TOZERO)
titles = ['Original Image','BINARY','TRUNC','TOZERO']
images = [img, thresh1, thresh3, thresh4]
for i in range (4):
    cv2.imshow(titles[i], images[i])
    print("-----------------------------------------------")
    print(titles[i])
    text = pytesseract.image_to_string(images[i])
    print(text)
    print()
    cv2.waitKey(0)
cv2.destroyAllWindows()
