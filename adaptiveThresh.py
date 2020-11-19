import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

image1 = cv2.imread("/home/nishka/Desktop/CV_Project/deblurred.png")  
   
# cv2.cvtColor is applied over the 
# image input with applied parameters 
# to convert the image in grayscale  
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
   
  
thresh2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,229,18)
 
# the window showing output images 
# with the corresponding thresholding  
# techniques applied to the input image 
cv2.imshow('Adaptive Gaussian', thresh2) 
  
text = pytesseract.image_to_string(thresh2)
print(text)
     
# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  

# thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY ,41,3)
# cv2.imshow("AT", thresh1)
# print("-----------------------------------------------")
# text = pytesseract.image_to_string(thresh1)
# print(text)
# print()
# cv2.waitKey(0)