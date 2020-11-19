import math
import numpy as np
import cv2
from matplotlib import pyplot as plt


threshold_values = {}

def calc_weight(s, e):
    w = 0
    for i in range(s, e):
        w += hist[i]
    return w

def calc_mean(s, e):
    m = 0
    w = calc_weight(s, e)
    for i in range(s, e):
        m += hist[i] * i
    return m/float(w)

def calc_variance(s, e):
    v = 0
    m = calc_mean(s, e)
    w = calc_weight(s, e)
    for i in range(s, e):
        # vairance put formula from notes 
        v += ((i - m) **2) * hist[i]
    v /= w
    return v

img = cv2.imread('samples/sample01.png', 0)

# CREATE HISTOGRAM
hist = np.zeros(256)
for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        hist[img[i,j]] += 1
x = np.arange(0,256)
plt.bar(x, hist, color='b', width=5, align='center', alpha=0.25)
plt.show()


for i in range(1, len(hist)):
    lower_var = calc_variance(0, i)
    lower_weight = calc_weight(0, i) / float(img.shape[0]*img.shape[1])
    
    upper_var = calc_variance(i, len(hist))
    upper_weight = calc_weight(i, len(hist)) / float(img.shape[0]*img.shape[1])
    
    intraclass_var = lower_weight * (lower_var) + upper_weight * (upper_var)
    
    if not math.isnan(intraclass_var):
        threshold_values[i] = intraclass_var

min_intra_var = min(threshold_values.values())
optimal_threshold = [key for key, val in threshold_values.items() if val == min_intra_var]
print('optimal threshold', optimal_threshold[0])


for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        if img[i,j] >= optimal_threshold[0]:
            img[i,j] = 255
        else:
            img[i,j] = 0


cv2.imshow("OTSU", img)
cv2.waitKey(0)

