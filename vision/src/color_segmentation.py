import cv2
import imutils
import numpy as np
import pdb
from homography import Homography

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def color_seg(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	low = (1, 190, 100) # dark red blocks
	high = (18, 255, 255) # bright orange cone
	thresholded = cv2.inRange(hsv, low, high)

	img2, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	box1 = ((0, 0), (0, 0))
	box2 = ((0, 0), (0, 0))
	boxes = list()
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		area = w*h
		boxes.append((area, ((x,y), (x+w, y+h))))

	boxes = sorted(boxes, reverse=True)
	box1 = boxes[0][1] # biggest box
	box2 = boxes[1][1] # second biggest box

	return box1
