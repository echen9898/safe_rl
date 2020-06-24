#!/usr/bin/env python2

# Standard imports
import os
import rospy
import numpy as np
from math import pi, cos, sin, atan, degrees, radians
import random
import rospkg
import csv

# Gym imports
import gym
import learning_gym

# ROS message imports
from std_msgs.msg import Int8, String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion, PolygonStamped, PoseWithCovarianceStamped, PointStamped
from tf.transformations import euler_from_quaternion

class Recorder:

    def __init__(self):
        
        # Pub/Sub
        self.sub_location     = rospy.Subscriber('/pf/viz/inferred_pose', PoseStamped, self.location_callback)
        self.sub_record       = rospy.Subscriber('/record_data', Int8, self.record_callback)
        self.sub_drive        = rospy.Subscriber('/last_action', Int8, self.drive_callback)

        # Trackers
        self.goal = np.array([2.86, 0.0])
	self.object = np.array([1.53, 0.0])
        self.state = np.array([2.86, 0.0, 0.0]) # [dx, dy, dheading]
        self.last_action = -1 # last action taken
        self.last_state = np.array([None, None]) # state directly after last action
        self.location = np.array([0.0, 0.0, 0.0]) # [x, y, heading]
	
	# Data
	self.file = 'p1.csv'
    	self.file_path = os.path.join(rospkg.RosPack().get_path('learning'), 'csv', self.file)
    	self.record_data = False
	self.vision_bound = 1.0472 # 60 degrees
	
    def record_callback(self, switch):
	   self.record_data = True
    
    def test_vision(self, heading):
	return -self.vision_bound <= abs(heading) <= self.vision_bound

    def location_callback(self, msg):
        heading = euler_from_quaternion([0, 0, msg.pose.orientation.z, msg.pose.orientation.w])[2]
        self.location = np.array([msg.pose.position.x, msg.pose.position.y, heading])
        dx, dy, dheading = self.get_state(self.location[0], self.location[1], self.location[2], self.goal)
	dx1, dy1, dheading1 = self.get_state(self.location[0], self.location[1], self.location[2], self.object)

	goal_in_sight = self.test_vision(dheading)
	object_in_sight = self.test_vision(dheading1)

	if goal_in_sight and not object_in_sight: # goal is closer
        	self.state = np.array([dx, dy, dheading])
	elif object_in_sight and not goal_in_sight: # object is closer
		self.state = np.array([dx1, dy1, dheading1])
	else:
		d_goal = (dx)**2 + (dy)**2
		d_object = dx1**2 + dy1**2
		if d_goal <= d_object:
			self.state = np.array([dx, dy, dheading])
		elif d_object < d_goal:
			self.state = np.array([dx1, dy1, dheading1])

    def drive_callback(self, action):
        self.last_action = action.data
        rospy.sleep(1.0)
        self.last_state = self.state

    def get_state(self, x, y, heading, target):

        # Determine heading coordinates (centered at robot location)
        right = radians(90)
        straight = radians(180)
        if -right < heading <= 0: # top right
            temp = right + heading
        elif -straight < heading <= -right: # bottom right
            temp = heading + right
        elif 0 < heading <= right: # top left
            temp = right + heading
        elif right < heading <= straight: # bottom left
            temp = - right - straight + heading 
        yh = -cos(temp) # assume length 1 heading vector
        xh = sin(temp)

        # Determine rotated heading coordinates, and change in heading required
        dx = target[0] - x # top side = positive, bottom side = positive
        dy = target[1] - y # right side = positive, left side = negative
        dheading =  0

        if y <= 0: # right half
            phi = atan(dx/dy)
            theta = right - phi # angle to translate by (0 to 90)
            yr = cos(theta)*yh - sin(theta)*xh
            xr = sin(theta)*yh + cos(theta)*xh
            heading_rot = atan(xr/-yr)
            if xr >= 0: # top side
                if heading_rot >= 0: # right side
                    dheading = right - heading_rot
                else: # left side
                    dheading = -(right + heading_rot)
            else: # bottom side
                if heading_rot >= 0: # left side
                    dheading = - (right + heading_rot) # positive
                else: # right side
                    dheading = right - heading_rot

        elif y > 0: # left half
            phi = atan(dx/-dy)
            theta = right - phi # angle to translate by (0 to 90)
            yr = cos(theta)*yh + sin(theta)*xh
            xr = -sin(theta)*yh + cos(theta)*xh     
            heading_rot = atan(xr/yr)
            if xr >= 0: # top side
                if heading_rot >= 0: # right side
                    dheading = -right + heading_rot
                else: # left side
                    dheading = (right + heading_rot)
            else: # bottom side
                if heading_rot >= 0: # left side
                    dheading = (right + heading_rot) # positive
                else: # right side
                    dheading = -right + heading_rot

        return dx, dy, dheading

    def record(self):
        print('RECORD: ', self.last_state, self.last_action, self.state)
        with open(self.file_path, 'a') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([str(self.last_state), self.last_action, str(self.state)])
            file.close()
            self.record_data = False

if __name__ == "__main__":
    rospy.init_node('recorder')
    recorder = Recorder()

    r = rospy.Rate(3)
    while not rospy.is_shutdown():
	if recorder.record_data:
		recorder.record()
    	r.sleep()

    rospy.spin()
