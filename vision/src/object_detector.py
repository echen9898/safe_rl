#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Image
from vision.msg import bounding_box
from color_segmentation import color_seg, image_print

from homography import Homography

class ObjectFinder:

	def __init__(self):

		# Pub/Sub
		self.sub_camera       = rospy.Subscriber('/zed/rgb/image_rect_color', Image, self.sub_callback_image)
		self.pub_bounding_box = rospy.Publisher('/bounding_box', bounding_box, queue_size = 10)
		self.pub_state        = rospy.Publisher('/state', Float32MultiArray, queue_size = 10)
		self.pub_image        = rospy.Publisher('/image_bounded', Image, queue_size = 10)
		
		# Image data
		self.bridge = CvBridge()
		self.H = Homography()

	def sub_callback_image(self, data):
		img = self.bridge.imgmsg_to_cv2(data, 'bgr8')

		box = color_seg(img)
		msg = bounding_box()
		
		msg.x_min = box[0][0]
		msg.y_min = box[0][1]
		msg.x_max = box[1][0]
		msg.y_max = box[1][1]
		
		dx, dy = self.H.find_box_bottom_center(box)
		print('BOX LOCAL: ', dx, dy)
		state_msg = Float32MultiArray()
		state_msg.data = [dx, dy]
		self.pub_state.publish(state_msg)

		img_new = img
		cv2.rectangle(img_new, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0,255,0), 2)
		msg_img = self.bridge.cv2_to_imgmsg(img_new, 'bgr8')
		
		self.pub_image.publish(msg_img)

if __name__ == '__main__':
	rospy.init_node('object_detector')
	finder = ObjectFinder()
	rospy.spin()
