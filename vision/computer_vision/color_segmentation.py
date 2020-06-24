import cv2
import imutils
import numpy as np
import pdb

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

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	low = (1, 150, 160)
	high = (40, 255, 255)
	thresholded = cv2.inRange(hsv, low, high)

	img2, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	bounding_box = ((0, 0), (0,0))
	max_area = 0
	x_best, y_best, w_best, h_best = 0,0,0,0
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		area = w*h
		if area > max_area:
			x_best, y_best, w_best, h_best = x, y, w, h
			bounding_box = ((x,y), (x + w, y + h))
			max_area = area
	cv2.rectangle(img, (x_best, y_best), (x_best+w_best, y_best+h_best), (0,255,0), 2)
	contoured = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

	cv2.waitKey()

	return bounding_box

def color_seg(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	low = (1, 150, 160)
	high = (40, 255, 255)
	thresholded = cv2.inRange(hsv, low, high)
	
	img2, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	bounding_box = ((0, 0), (0,0))
	max_area = 0
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		area = w*h
		if area > max_area:
			bounding_box = ((x,y), (x + w, y + h))
			max_area = area
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	contoured = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
	return bounding_box
