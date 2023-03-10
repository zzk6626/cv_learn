import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 模板匹配
img = cv.imread('./image/wulin.jpeg')
plt.imshow(img[:,:,::-1])
plt.show()

template = cv.imread('./image/bai.jpeg')
plt.imshow(template[:,:,::-1])
plt.show()

res = cv.matchTemplate(img,template,cv.TM_CCORR)  # 指标选择 相关匹配，数值越高表示匹配越完美,左上角开始当做定位，放置图片匹配
plt.imshow(res,cmap=plt.cm.gray)
plt.show()

min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)   # 确定最匹配的位置和值
top_left = max_loc
h,w = template.shape[:2]
bottom_right = (top_left[0]+w,top_left[1]+h)

cv.rectangle(img,top_left,bottom_right,(0,255,0),2)  # 颜色(0,255,0), 线的宽度2
plt.imshow(img[:,:,::-1])
plt.show()
# 模板匹配不适用于尺度变换，视角变换后的图像，这时我们就要使用关键点匹配算法，比较经典的关键点检测算法包括SIFT和SURF等
# 主要的思路是首先通过关键点检测算法获取模板和测试图片中的关键点；然后使用关键点匹配算法处理即可，这些关键点可以很好的处理尺度变化、视角变换、旋转变化、光照变化等，具有很好的不变性。

# 霍夫变换
# 对偶性 x,y平面上一条直线 ——> k,b平面一个点 ///  k,b平面上一条直线 ——> x,y平面一个点
# 由于k,b平面上 x=2直线上的点无法求解(在k,b平面上平行，无交点)，引申到r,sita 极坐标参数平面，同样对偶性成立,r=x*cos(sita)+y*sin(sita)
# 图像空间的一个点在参数空间中对应为一条曲线，参数空间的每个点都对应图像空间的一条直线
# 如果一幅图像中的像素构成一条直线，那么这些像素坐标值（x, y）在参数空间对应的曲线一定相交于一个点，所以我们只需要将图像中的所有像素点（坐标值）变换成参数空间的曲线，并在参数空间检测曲线交点就可以确定直线
# 利用上述原理，从原点开始旋转，将sita离散化，统计每个点(x,y)到直线的r，进而统计满足同一sita和r的点数目，进入累加器，进而确定r,sita，保证经过最多的同一类像素点。

# 霍夫线检测
# 先进行边缘检测，在霍夫检测
img = cv.imread('./image/rili.jpg')
gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)  # 灰度值形式读入

edges = cv.Canny(gray,50,150)
plt.imshow(edges,cmap=plt.cm.gray)
plt.show()

lines = cv.HoughLines(edges,0.8,np.pi/18,150)  # 0.8--r精度  pi/180--sita精度  累加器阈值-150
print(lines)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b
    # 延长点位置线
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,255,0))

plt.imshow(img[:,:,::-1])
plt.show()

# 霍夫圆检测
planets = cv.imread('./image/star.jpeg')
# 读取灰度图像
gray_img = cv.cvtColor(planets,cv.COLOR_RGB2GRAY)
# 进行中值模糊，去噪点
img = cv.medianBlur(gray_img,7)  # 核的大小
plt.imshow(img,cmap=plt.cm.gray)
plt.show()
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,200,param1=100,param2=50,minRadius=0,maxRadius=100)
print(circles)  # 前两个圆心，第三个是半径

for circle in circles[0,:]:
    cv.circle(planets,(int(circle[0]),int(circle[1])),int(circle[2]),(0,255,0),2)   # 圆
    cv.circle(planets,(int(circle[0]),int(circle[1])), 2, (0, 255, 0),3)  # 绘制圆心/半径为2/  -1->填充起来

plt.imshow(planets[:,:,::-1])
plt.show()
