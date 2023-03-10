import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((512,512,3),np.uint8)
cv.imwrite('data/hei.jpg',img)  # 保存图像

cv.line(img,(0,0),(511,511),(255,0,0),5)   # opencv里面是BGR 所以(255,0,0)代表蓝色   RGB--红绿蓝,5--线性宽度
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.circle(img,(447,63),63,(0,0,225),-1)  # -1 -- 填充 ， 4 -- 线条宽度

font = cv.FONT_HERSHEY_SIMPLEX   # 字体
cv.putText(img,'Opencv',(10,500),font,4,(255,255,255),2,cv.LINE_AA)  # 4-字体的大小，线形--cv.LINE_AA，2-字体线条的宽度

plt.imshow(img[:,:,::-1])  # 按照GRB显示
plt.title('匹配结果'),plt.xticks([]),plt.yticks([])
plt.show()