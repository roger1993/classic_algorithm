# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
from matplotlib import pyplot as plt
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
 
# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
#plt.title("template")
#plt.imshow(template)
#plt.show()
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#cv2.imshow("template", template)
#cv2.waitKey(0)
#template = cv2.Canny(template, 50, 200)
(h, w) = template.shape[:2]
#print(h,w)
#plt.subplot(121)
#plt.imshow(template, cmap = "gray", interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])

with open ("/Users/roger/Downloads/demo/输出/msd_output/result.txt", "w") as f:
# loop over the images to find the template in
	counter = 1
	for imagePath in glob.glob(args["images"] + "/*.*"):
		# load the image, convert it to grayscale, and initialize the
		# bookkeeping variable to keep track of the matched region
		image = cv2.imread(imagePath)
		#cv2.imshow("img", image)
		#cv2.waitKey(0)
		#image = cv2.resize(image, (512,512))
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#image_gray = cv2.Canny(image_gray, 50, 200)

		res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
		threshold = 0.60
		loc = np.where(res >= threshold)
		#print(len(res[loc])) 
		if len(res[loc]) <= 300:
			f.write("{} 存在缺陷！".format(imagePath.split("/")[-1]))
			#print("{} 存在缺陷！".format(imagePath.split("/")[-1]))
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

			top_left = max_loc
			bottom_right = (top_left[0] + w, top_left[1] + h)
			cv2.rectangle(image,top_left, bottom_right, (0,0,255), 2)
			cv2.imwrite("/Users/roger/Downloads/demo/输出/msd_output/" + "缺陷样本" + str(counter) + ".bmp", image)
			counter += 1
		else:
			counter += 1
			continue
