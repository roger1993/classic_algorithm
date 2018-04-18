import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import glob
from skimage.measure import compare_ssim


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize", default=True,
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())


f = open("/Users/roger/Downloads/demo/输出/划痕_output/result.txt", "w")
fcount = 0
for imagePath in glob.glob(args["image"] + "/*.bmp"):
	img = cv2.imread(imagePath)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#plt.imshow(img_gray, cmap="gray")
	#plt.show()
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	img_gray = clahe.apply(img_gray)
	#plt.imshow(img_gray, cmap="gray")
	#plt.show()
	#30,30
	blur = cv2.blur(img_gray,(21,21))
	#blur = cv2.GaussianBlur(,(5,5),0)
	#diff = cv2.subtract(img_gray, blur)
	#score, diff = compare_ssim(img_gray, blur, full=True)
	#diff = (diff * 255).astype("uint8")
	diff = cv2.absdiff(img_gray, blur)

	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	# diff = clahe.apply(diff)
	#plt.imshow(diff, cmap="gray")
	#plt.show()
	#ret, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	ret, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
	#_, mask = cv2.connectedComponents(mask)
	#plt.imshow(mask, cmap="gray")
	#plt.show()
	#plt.imshow(mask, cmap="gray")
	#plt.show()
	#cv2.threshold(diff, 50, 255,cv2.THRESH_BINARY)
	#ret, mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	#cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('mask', 1000, 1000)
	#cv2.imshow("mask", diff)
	#cv2.imshow("imagemean", blur)
	#cv2.imshow("diff", diff)
	#cv2.imshow("Mask", mask)
	#cv2.waitKey(0)

	#print(mask.shape)



	#imagegray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	#11,17,17
	#9,75,75
	#11, 18, 18
	#mask = cv2.bilateralFilter(mask, 40, 5, 5)
	edged = cv2.Canny(mask, 10, 150)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	edged = cv2.dilate(edged, kernel, iterations=1)
	#plt.imshow(edged, cmap="gray")
	#plt.show()
	im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	drawing = np.zeros(mask.shape, dtype="uint8")

	for i in range(len(contours)):
		cnt = contours[i]
		M = cv2.moments(cnt)
		area = M['m00']

		#2000-9000
		if area > 10 and area < 9999:
			cv2.drawContours(drawing, [cnt], -1, 255, 3)

	#plt.imshow(drawing, cmap="gray")
	#plt.show()

	#drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	close = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, kernel)
	#opening = cv2.morphologyEx(drawing, cv2.MORPH_OPEN, kernel)
	#print(1)
	#plt.imshow(close, cmap="gray")
	#plt.show()
	#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	#close = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
	#cv2.namedWindow("close", cv2.WINDOW_NORMAL)
	#cv2.resizeWindow('close', 1000, 1000)
	#cv2.imshow("close", close)
	#cv2.waitKey(0)

	#！！！！ drawing
	im2, contours1, hierarchy1 = cv2.findContours(close, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	counter = 0
	for i in range(len(contours1)):
		cnt1 = contours1[i]
		# print(cnt1)
		# break
		M = cv2.moments(cnt1)
		area = M['m00']

		#5000-100000
		if area > 2000 and area < 6000:
			cv2.drawContours(img, [cnt1], -1, (0,0,255), 3)
			counter += 1

		# elif area > 1200 and area < 2000:
		# 	cv2.drawContours(img, [cnt1], -1, (255,0,0), 3)
		# 	counter += 1

	cv2.imwrite("/Users/roger/Downloads/demo/输出/划痕_output"+ str(fcount) + ".bmp", img)		
	if counter >= 1:
			#print("NG !!!")
		print("{} is NG！!".format(imagePath.split("/")[-1]))
		f.write("{} is NG！!\n".format(imagePath.split("/")[-1]))
			#print("注意，存在掉漆或者划痕！")

	else:
			#print("OK !!!")
		print("{} is OK！!".format(imagePath.split("/")[-1]))
		f.write("{} is OK！!\n".format(imagePath.split("/")[-1]))
				# elif area >=0 and area <= 400:
				# 	cv2.drawContours(img, [cnt1], 0, (255,0,0), 3)
	fcount += 1
	#cv2.imshow("")
	if args["visualize"]:
		cv2.namedWindow("img", cv2.WINDOW_NORMAL)
		cv2.resizeWindow('img', 1000, 1000)
		cv2.imshow("img", img)
		cv2.waitKey(0)
f.close()
