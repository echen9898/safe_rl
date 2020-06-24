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
        self.sub_drive_commands = rospy.Subscriber('/drive_commands', Int8, self.drive)
        self.pub_drive = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)

        self.velocity = 10.0 # meters/second
        self.directions = {
        0 : ['FORWARD', [[0.0, self.velocity, 0.06], [0.0, 0.0, 0.1]]],
        1 : ['LEFT', [[0.34, self.velocity, 0.06], [0.0, 0.0, 0.1]]],
        2 : ['RIGHT', [[-0.34, self.velocity, 0.06], [0.0, 0.0, 0.1]]],
        3 : ['NO_MOVE', [[0.0, 0.0, 5.0]]]
        }

    def drive(self, angle_index):

        direction, move_sequence = self.directions[angle_index.data]
        # print('DIRECTION: ' + direction)

        for move in move_sequence:
            angle, velocity, duration = move
            
            command = AckermannDriveStamped()
            command.drive.steering_angle = angle
            command.drive.speed = velocity

            end_time = rospy.Time.now() + rospy.Duration(duration)
            while rospy.Time.now() <= end_time:
                self.pub_drive.publish(command)


if __name__ == "__main__":
    rospy.init_node('drive')
    driver = drive_control()
    rospy.spin()