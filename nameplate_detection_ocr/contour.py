import cv2
import numpy as np
import imutils
import pytesseract
from PIL import Image
import re
import argparse
import os

def order_points(pts):
  rect = np.zeros((4, 2), dtype = "float32")
  s = pts.sum(axis = 1)
  #print(np.argmin(s)-1)
  #print(np.argmax(s)-1)
  #print(s)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  return rect

def four_point_transform(image, pts):
  rect = order_points(pts)
  (tl, tr, br, bl) = rect

  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
 
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
 
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
 
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  return warped


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
  help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize", default=False,
  help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
#cv2.waitKey(0)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("bi", gray)
#cv2.waitKey(0)
edged = cv2.Canny(gray, 30, 200)

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
  # approximate the contour
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.02 * peri, True)

 
  # if our approximated contour has four points, then
  # we can assume that we have found our screen
  if len(approx) == 4:
    screenCnt = approx

    #cv2.drawContours(image, [approx], -1, 255)
    #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('img', 1000, 1000)
    #cv2.imshow("img", image)
    #cv2.waitKey(0)
    break



warped = four_point_transform(gray, screenCnt.reshape(4, 2))
output = four_point_transform(image, screenCnt.reshape(4, 2))

#cv2.imshow("img", warped)
#cv2.waitKey(0)

thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
# loop over the digit area candidates
counter = 0
for c in cnts:
  # compute the bounding box of the contour
  (x, y, w, h) = cv2.boundingRect(c)
  #cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),2)
  if w >= 100 and h >= 50:
    roi = thresh[y:y + h, x:x + w]
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite("/Users/roger/Downloads/demo/输出/铭牌_output/"+str(counter)+".bmp", roi)
    counter+=1
    # img = Image.open(roi)
    # text = pytesseract.image_to_string(img, lang="chi_sim+eng")
    # print(text)

#cv2.imshow("img1", thresh)
#cv2.imshow("img", output)
#cv2.waitKey(0)

f1 = open("/Users/roger/Downloads/demo/输出/铭牌_output/ocr_result.txt", "w")
f2 = open("/Users/roger/Downloads/demo/输出/铭牌_output/result.txt", "w")
for imagePath in os.listdir("/Users/roger/Downloads/demo/输出/铭牌_output/"):
  if imagePath.endswith(".bmp"):
    print(imagePath)
#for imagePath in glob.glob("/Users/roger/Downloads/ningde/test/" + "test*" + ".*"):
    img = Image.open("/Users/roger/Downloads/demo/输出/铭牌_output/" + imagePath)
    text = pytesseract.image_to_string(img, lang="chi_sim+eng")
    print(text)
    f1.write(text + "\n")

    counter = 0
    for char in text:
    #print(char)
      if char < "\u4e00" or char > "\u9fff":
      #print(char)
        if not re.search('\w+', char):  
          counter += 1  
          #print("识别错误")
          #continue

    #print("counter is {}".format(counter))
    if counter >= 5:
    #print(counter)
      print("{} is NG !!!!".format(args["image"].split("/")[-1]))
      f2.write("{} is NG !!!!".format(args["image"].split("/")[-1]))
f1.close()
f2.close()


  
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# cv2.imshow("img", thresh)
# cv2.waitKey(0)

#cv2.drawContours(image, [approx], -1, 255)
# cv2.imshow("img", warped)
# cv2.waitKey(0)

#screenCnt = np.squeeze(screenCnt, axis=1)
# temp = screenCnt[1]
# screenCnt[1] = screenCnt[2]
# screenCnt[2] = temp
#print(screenCnt[1])
  



# mask = np.zeros_like(image)
# cv2.drawContours(mask, [screenCnt], -1, 255, -1) # Draw filled contour in mask
# out = np.zeros_like(image) # Extract out the object and place into output image
# out[mask == 255] = image[mask == 255]
# cv2.imshow('Output', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#screenCnt = np.squeeze(screenCnt, axis=1)
#print(screenCnt.shape)

# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
# cv2.imshow("Game Boy Screen", image)
# cv2.waitKey(0)

# Four corners of the book in destination image.
# pts_dst = np.array([[0, 0],[roi.shape[1], 0],[0, roi.shape[0]],[roi.shape[1], roi.shape[0]]])

# # Calculate Homography
# h, status = cv2.findHomography(screenCnt, pts_dst)
 
# # Warp source image to destination based on homography
# im_out = cv2.warpPerspective(roi, h, (roi.shape[1], roi.shape[0]))

# cv2.imshow("A",im_out)
# cv2.waitKey(0)

#cv2.imwrite("./test/hmg.bmp",im_out)