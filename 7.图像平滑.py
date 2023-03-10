import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 高斯滤波,适合处理高斯噪声
doggau = cv.imread('image/dogGauss.jpeg')
plt.imshow(doggau[:,:,::-1])
plt.show()
dog = cv.GaussianBlur(doggau,(3,3),1)
plt.imshow(dog[:,:,::-1])
plt.show()

# 中值滤波，适合处理椒盐噪声
dogsp = cv.imread('image/dogsp.jpeg')
plt.imshow(dogsp[:,:,::-1])
plt.show()
dog = cv.medianBlur(dogsp,3)
plt.imshow(dog[:,:,::-1])
plt.show()
