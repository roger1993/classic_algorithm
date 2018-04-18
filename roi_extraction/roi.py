# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
from matplotlib import pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Path to images where template will be matched")
args = vars(ap.parse_args())
 
im = cv2.imread(args["image"])
h,w = im.shape[:2]
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', w, h)
#im = cv2.resize(im, (1024,1024), cv2.INTER_AREA)
r = cv2.selectROI("Image", im)
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

short_cut = args["image"].split("/")[-1]
cv2.imwrite("./roi/" + str(short_cut), imCrop)


