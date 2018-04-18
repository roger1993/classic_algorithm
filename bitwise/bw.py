import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import glob

im_gray = cv2.imread('/Users/roger/Downloads/衡阳钢管/轧痕A_15/Image__2018-01-09__11-06-48.bmp', 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
im_gray = clahe.apply(im_gray)

(thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
h,w = im_gray.shape[:2]
cv2.namedWindow("bw", cv2.WINDOW_NORMAL)
cv2.resizeWindow('bw', w, h)
cv2.imshow("bw", im_bw)
cv2.waitKey(0)