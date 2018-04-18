import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Write some Text

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

cv2.putText(img,'Hello World!', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

#Display the image
cv2.imshow("img",img)
cv2.waitKey(0)

#Save image
cv2.imwrite("out.jpg", img)
