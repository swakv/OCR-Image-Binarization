import cv2
import numpy as np
import math
import scipy.ndimage
import pytesseract

imgname = '../samples/sample01.png'
filt_radius = 25
k_threshold = 0.001
X = cv2.imread(imgname, 0)
X = X.astype('float32')
X /= 255.0

fgrid = list(range(-filt_radius, filt_radius+1))

x,y = np.meshgrid(fgrid, fgrid)

sums = x**2 + y**2
sums_sq = np.sqrt(sums)
filt = sums_sq <= filt_radius

filt = filt / filt.sum()

local_mean = scipy.ndimage.correlate(X, filt, mode='constant')
local_std = np.sqrt(scipy.ndimage.correlate(X**2, filt, mode='constant'))

X_bin = (X >= (local_mean + k_threshold * local_std))


X_bin = X_bin.astype(int)

cv2.imshow("final", X_bin)
cv2.waitKey(0)

text = pytesseract.image_to_string(X_bin)
print(text)

