#!/usr/bin/env python2

# Standard imports
import rospy
import numpy as np
import math
import random

# Gym imports
import gym
import learning_gym

# ROS message imports
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion, PolygonStamped, PoseWithCovarianceStamped, PointStamped

class learner:

    def __init__(self):
        self.env = gym.make('Learning-v0')

if __name__ == "__main__":
    rospy.init_node('learning')

    agent = learner()
    agent.env.reset()

    r = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        agent.env.step(random.randint(0, 3))
        r.sleep()

    rospy.spin()