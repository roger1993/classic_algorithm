import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import glob

MIN_MATCH_COUNT = 200

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
	help="Path to template image")

ap.add_argument("-i", "--images", required=True,
	help="Path to images where template will be matched")

args = vars(ap.parse_args())
 
img1 = cv2.imread(args["query"])
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

with open ("/Users/roger/Downloads/demo/输出/msd_output/result.txt", "w") as f:
	for imagePath in glob.glob(args["images"] + "/*.bmp"):
		img2 = cv2.imread(imagePath)
		gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		sift = cv2.xfeatures2d.SIFT_create()
		kp1, des1 = sift.detectAndCompute(gray_img1, None)
		kp2, des2 = sift.detectAndCompute(gray_img2, None)

		# bf = cv2.BFMatcher()
		# matches = bf.knnMatch(des1,des2, k=2)

		# # Apply ratio test
		# good = []
		# for m,n in matches:
		# 	if m.distance < 0.85*n.distance:
		# 		good.append([m])

		# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

		# plt.imshow(img3)
		# plt.show()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)   # or pass empty dictionary

		flann = cv2.FlannBasedMatcher(index_params,search_params)

		matches = flann.knnMatch(des1,des2,k=2)

		# Need to draw only good matches, so create a mask
		matchesMask = [[0,0] for i in range(len(matches))]

		# ratio test as per Lowe's paper
		counter = 0
		for i,(m,n) in enumerate(matches):
			#print(m.distance)
			#print("\n")
			#print(n.distance)
			if m.distance < 0.75*n.distance:
				matchesMask[i]=[1,0]
				counter += 1

		draw_params = dict(matchColor = (0,255,0),
		                   singlePointColor = (255,0,0),
		                   matchesMask = matchesMask,
		                   flags = 0)

		img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(img3)
		plt.show()