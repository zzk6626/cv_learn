# 实时检测
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Fast算法  <并没有加入机器学习的方法，但可以添加非极大值抑制>

# 优化方法，机器学习
# 检索图像中的每一个特征点 ---->  一个特征点  --> 周围的16个像素组成一个向量P
# 每个特征点的16像素属于下列三类中的一种 特征向量P -->  三个子集 (pd,ps,pb)
# 定义一个新的布尔变量Kp，如果p是角度，设置为True，不是则为False
# 利用特征值向量P，目标值为kp，训练决策树分类器
# 将构建好的决策树运用于其他图像的快速的检测

# 优化方法，非极大值抑制
# 在筛选出来的候选角点中很多是紧凑在一起的，需要通过非极大值抑制消除这种影响。
# 所有候选角点确定一个打分函数V
# 比较相邻候选角点的V值，把较小的候选点pass掉

# 1 读取图像
img = cv.imread('./image/tv.jpg')
# 2 Fast角点检测
# 2.1 创建一个Fast对象，传入阈值，注意：可以处理彩色空间图像

# 设置非极大值抑制的两种办法：直接在threshold=30后加入nonmaxSuppression=0，或者在创建完成后fast.setNonmaxSuppression(0)
fast = cv.FastFeatureDetector_create(threshold=30)

# 2.2 检测图像上的关键点
kp = fast.detect(img,None)
# 2.3 在图像上绘制关键点
img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255))

# 2.4 输出默认参数
print( "Threshold: {}".format(fast.getThreshold()))
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )


# 2.5 关闭非极大值抑制
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
# 2.6 绘制为进行非极大值抑制的结果
img3 = cv.drawKeypoints(img, kp, None, color=(0,0,255))

# 3 绘制图像
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img2[:,:,::-1])
axes[0].set_title("加入非极大值抑制")
axes[1].imshow(img3[:,:,::-1])
axes[1].set_title("未加入非极大值抑制")
plt.show()
