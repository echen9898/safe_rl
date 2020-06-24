#!/usr/bin/env python2

# Standard imports
import rospy
import numpy as np
import math

# Gym imports
import gym
import learning_gym

# ROS message imports
from std_msgs.msg import Int8
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray

class drive_control:

    def __init__(self):

        # Sub/Pub
        self.sub_drive_commands = rospy.Subscriber('/drive_commands', Int8, self.set_action)
        self.pub_drive = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)
	self.pub_last_action = rospy.Publisher('/last_action', Int8, queue_size=20)
	
	# Info
        self.velocity = 0.7 # meters/second
	self.action = None
        self.directions = {
        0 : ['FORWARD', [[0.0, self.velocity, 1.3], [0.0, 0.0, 1.5]]],
        1 : ['LEFT', [[0.34, self.velocity, 1.0], [0.0, 0.0, 1.5]]],
        2 : ['RIGHT', [[-0.34, self.velocity, 1.0], [0.0, 0.0, 1.5]]],
        }
    
    def set_action(self, action):
        self.action = action

    def drive(self, angle_index):
        direction, move_sequence = self.directions[angle_index.data]

        for move in move_sequence:
            angle, velocity, duration = move
            
            command = AckermannDriveStamped()
            command.drive.steering_angle = angle
            command.drive.speed = velocity

            end_time = rospy.Time.now() + rospy.Duration(duration)
            while rospy.Time.now() <= end_time:
                self.pub_drive.publish(command)
	
    	self.pub_last_action.publish(angle_index)
    	self.action = None # wait for next command

if __name__ == "__main__":
    rospy.init_node('drive')
    driver = drive_control()

    r = rospy.Rate(3)
    while not rospy.is_shutdown():
        if driver.action != None:
		driver.drive(driver.action)
        r.sleep()

    rospy.spin()
