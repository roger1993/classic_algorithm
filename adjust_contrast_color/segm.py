import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread("/Users/roger/Downloads/衡阳钢管/轧痕A_15/Image__2018-01-09__11-06-48.bmp",0)
# plt.imshow(img, cmap="gray")
# plt.show()

# ret,thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# plt.imshow(thresh1, cmap="gray")
# plt.show()

# kernel = np.ones((20,20),np.uint8)
# opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
# plt.imshow(opening, cmap="gray")
# plt.show()

# kernel = np.ones((120,120),np.uint8)
# erosion = cv2.erode(opening,kernel,iterations = 1)
# plt.imshow(erosion, cmap="gray")
# plt.show()

# merged = cv2.bitwise_and(img, img , mask=erosion)
# plt.imshow(merged, cmap="gray")
# plt.show()






# for the contrast

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

img = cv2.imread('/Users/roger/Downloads/宁德数据集/CATL样本图/顶面左侧/右侧开启（顶面左）.bmp')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax1.set_title("Before Adjust")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(img1)
#cv2.imshow("hey", img)
#cv2.waitKey(0)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.imshow("img", img)
# cv2.waitKey(0)
#w,h = img.shape[0], img.shape[1]
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8,8))
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
cl1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
#cl2 = cv2.cvtColor(cl1, cv2.COLOR_RGB2GRAY)
#ret,thresh1 = cv2.threshold(cl2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#sobelx = cv2.Sobel(cl2, cv2.CV_64F,0,1,ksize=-1)#默认ksize=3
#abs_sobel64f = np.absolute(sobelx)
#sobel_8u = np.uint8(abs_sobel64f)
# #sobelx = cv2.Laplacian(img, cv2.CV_64F)

ax2.set_title("After Adjust")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(cl1)
plt.show()

# cv2.namedWindow("cl1", cv2.WINDOW_NORMAL)
# #cv2.resizeWindow('cl1', w, h)
# cv2.imshow("cl1", cl1)
# cv2.waitKey(0)
# cv2.imwrite('clahe_2.jpg',cl1)


#for the gradient image

# img = cv2.imread('/Users/roger/Downloads/衡阳钢管/轧痕A_15/Image__2018-01-09__11-06-48_sc.bmp',0)
# sobelx = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)#默认ksize=3
# abs_sobel64f = np.absolute(sobelx)
# sobel_8u = np.uint8(abs_sobel64f)
# #sobelx = cv2.Laplacian(img, cv2.CV_64F)
# #edges = cv2.Canny(img,30,300)
# cv2.namedWindow("edge", cv2.WINDOW_NORMAL)
# cv2.imshow("edge", sobel_8u)
# cv2.waitKey(0)