# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
from matplotlib import pyplot as plt
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
	help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
 
# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
#plt.title("template")
#plt.imshow(template)
#plt.show()
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#template = cv2.Canny(template, 50, 200)
(h, w) = template.shape[:2]
#plt.subplot(121)
#plt.imshow(template, cmap = "gray", interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])

with open ("/Users/roger/Downloads/demo/输出/角件_output/result.txt", "w") as f:
# loop over the images to find the template in
	counter = 1
	for imagePath in glob.glob(args["images"] + "/*.*"):
	# load the image, convert it to grayscale, and initialize the
	# bookkeeping variable to keep track of the matched region
		image = cv2.imread(imagePath)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image_gray = cv2.Canny(image_gray, 50, 200)
		res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
		threshold = 0.6
		loc = np.where(res >= threshold)
		#print(len(res[loc]))
		if len(res[loc]) <= 5:
			f.write("{} is NG !!\n".format(imagePath.split("/")[-1]))
			#print("{}存在缺陷！".format(imagePath.split("/")[-1]))
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

			top_left = max_loc
			bottom_right = (top_left[0] + w, top_left[1] + h)
			cv2.putText(image, "NG", (100,0), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
			cv2.rectangle(image,top_left, bottom_right, (0,0,255), 2)
			cv2.imwrite("/Users/roger/Downloads/demo/输出/角件_output/" + "缺陷样本" + str(counter) + ".bmp", image)
			counter += 1
		else:
			counter += 1
			continue
