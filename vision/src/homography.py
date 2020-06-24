#!/usr/bin/env python
import rospy
import math
import numpy as np
import cv2
from cv_bridge import CvBridge

from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from vision.msg import bounding_box, object_distance

class Homography():
	"""
	Calculates a homography matrix given reference points, and
	can be used to calculate an objects position given an image.
	"""
	def __init__(self):

		# Publishers and Subscribers
		# self.sub_box = rospy.Subscriber('/bounding_box', bounding_box, self.find_xy)
		# self.pub_distance = rospy.Publisher('/distance', object_distance, queue_size=10)

		# Homography data and matrix
		ground = np.array([
			(0.3048, -0.1524, 1),
			(0.3048, 0.1524, 1),
			(0.3048, 0.0, 1),
			(0.6096, -0.1524, 1),
			(0.6096, 0.1524, 1),
			(0.6096, 0.0, 1),
			(0.9144, -0.1524, 1),
			(0.9144, 0.1524, 1),
			(0.9144, 0.0, 1),
			(0.9144, -0.4572, 1),
			(0.6096, -0.4572, 1),
			(0.3048, -0.4572, 1),
			(0.6096, 0.4572, 1),
			(0.9144, 0.4572, 1)
			])
		self.ground_ref = np.float32(ground[:, np.newaxis, :])
		img = np.array([
			(522.0, 651.0, 1),
			(971.0, 656.0, 1),
			(744.0, 651.0, 1),
			(580.0, 525.0, 1),
			(849.0, 529.0, 1),
			(720.0, 529.0, 1),
			(604.0, 474.0, 1),
			(796.0, 476.0, 1),
			(702.0, 476.0, 1),
			(442.0, 472.0, 1),
			(351.0, 523.0, 1),
			(147.0, 638.0, 1),
			(1129.0, 529.0, 1),
			(996.0, 479.0, 1)
			])
		self.img_ref = np.float32(img[:, np.newaxis, :])
		self.H = cv2.findHomography(self.img_ref, self.ground_ref)[0]

	def find_xy(self, coord): # coord: [x, y]
		point = np.dot(self.H, np.array(coord))
		point /= point[2]
		return (point[0], -point[1])

	def find_box_bottom_center(self, box): # box: [(x0, y0), (x1, y1)] -> [top left, bottom right]
		x0 = box[0][0]
		x1 = box[1][0]
		y0 = box[0][1]
		y1 = box[1][1]
		center = np.array([(x1 - x0)/2 + x0, y1, 1]) #center base of bounding box
		point = self.find_xy(center)
		return (point[0], point[1])
