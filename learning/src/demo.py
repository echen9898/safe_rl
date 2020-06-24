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
from std_msgs.msg import Int8, String, FLoat32, Float32MultiArray
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion, PolygonStamped, PoseWithCovarianceStamped, PointStamped
from tf.transformations import euler_from_quaternion

class Demo:

    def __init__(self):
        
        # Pub/Sub
	    self.sub_location = rospy.Subscriber('/pf/viz/inferred_pose', PoseStamped, self.location_callback)
        self.sub_state = rospy.Subscriber('/state', Float32MultiArray, self.goal_callback)
        self.pub_drive = rospy.Publisher('/drive_commands', Int8, queue_size=10)

        # Environment Parameters
        self.goal = np.array([2.86, 0.0])
	    self.object = np.array([1.53, 0.0])
        self.vision_field_limit = radians(60)
        self.dx = 0.249
        self.dy = 0.198
        self.num_angles = 60
        self.feature_bins = [[_ for _ in np.arange(0.0, 3.0, self.dx)], [_ for _ in np.arange(-1.19, 1.19, self.dy)], [_ for _ in np.arange(-pi, pi, (2*pi)/self.num_angles)]]
        self.feature_bins_sim = [[_ for _ in np.arange(0.0, 3.0, self.dx)], [_ for _ in np.arange(-1.19, 1.19, self.dy)], [_ for _ in np.arange(-pi, pi, (2*pi)/8)]]

        # Trackers
        self.run = True # whether or not to run demo
        self.state = np.array([2.86, 0.0, 0.0]) # [dx, dy, dheading]
        self.discrete_state = np.array([0.0, 0.0, 0.0]) # discrete state
	    self.discrete_state_sim = np.array([0.0, 0.0, 0.0]) # discrete state, sim feature bins
        self.location = np.array([0.0, 0.0, 0.0]) # [x, y, heading]
        self.prob_thresh = 0.63 # probabilities above this are blind spots
	
    	# Data
    	self.blind_spot_file = 'v5_probs.csv' # blind spot file
    	self.blind_spot_file_path = os.path.join(rospkg.RosPack().get_path('learning'), 'csv/blind_spots', self.blind_spot_file)
        self.sim_policy_file = 'Q.csv' # sim policy file
        self.sim_policy_file_path = os.path.join(rospkg.RosPack().get_path('learning'), 'csv/sim/learn_policy_v7', self.sim_policy_file)
        self.blind_spots = dict() # {state : best_action}
        self.sim_policy = dict() # {state : probability}

    def location_callback(self, msg):
        heading = euler_from_quaternion([0, 0, msg.pose.orientation.z, msg.pose.orientation.w])[2]
        self.location = np.array([msg.pose.position.x, msg.pose.position.y, heading])
        dx, dy, dheading = self.get_state(self.location[0], self.location[1], self.location[2], self.goal)

        # dx1, dy1, dheading1 = self.get_state(self.location[0], self.location[1], self.location[2], self.goal)
    	# dx2, dy2, dheading2 = self.get_state(self.location[0], self.location[1], self.location[2], self.object)
    	
    	# goal_in_sight = self.test_vision(dheading1)
    	# object_in_sight = self.test_vision(dheading2)
    	# if goal_in_sight and not object_in_sight: # goal is closer
    	# 	self.state = np.array([dx1, dy1, dheading1])
    	# elif object_in_sight and not goal_in_sight: # object is closer
    	# 	self.state = np.array([dx2, dy2, dheading2])
    	# else:
    	# 	d_goal = dx1**2 + dy1**2
    	# 	d_object = dx2**2 + dy2**2
    	# 	if d_goal <= d_object:
    	# 		self.state = np.array([dx1, dy1, dheading1])
    	# 	elif d_object < d_goal:
    	# 		self.state = np.array([dx2, dy2, dheading2])

    def goal_callback(self, msg):
        dx, dy = msg.data
        x_goal = self.location[0] + dx
        y_goal = self.location[1] + dy
        self.goal = np.array([x_goal, y_goal])

    def set_state(self):
        s = self.state
        sd = list()
	    sd_sim = list()
        for i in range(3):
            closest = min(self.feature_bins[i], key=lambda x:abs(s[i]-x))
    	    closest_sim = min(self.feature_bins_sim[i], key=lambda x:abs(s[i]-x))
    	    sd_sim.append(closest_sim)
            sd.append(closest)
        self.discrete_state = np.array(sd)
	    self.discrete_state_sim = np.array(sd_sim)

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

    def test_termination(self):
        dx, dy, dheading = self.state
        if (0 <= dx <= 0.4) and (-0.3 <= dy <= 0.3):
            print('REACHED GOAL!')
            self.run = False

    def test_vision(self, heading):
	    return -self.vision_field_limit <= abs(heading) <= self.vision_field_limit

    def generate_blind_spots(self):
        with open(self.blind_spot_file_path) as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                x = (float(row[0][1:]), float(row[1]), float(row[2][:-1]))
                self.blind_spots[x] = float(row[3])

    def generate_sim_policy(self):
        with open(self.sim_policy_file_path) as file:
            reader = csv.reader(file, delimiter=',')
            line = 0
            actions = list()
            for row in reader:
                actions.append((float(row[4]), int(row[3])))
                if line%3 == 2: # last action
                    actions.sort()
                    x = (float(row[0][1:]), float(row[1]), float(row[2][:-1]))
                    self.sim_policy[x] = actions[-1][1]
		    if line == 1332:
			print(actions)
                    actions = list()
                line += 1

    def determine_action(self):
	key = (round(self.discrete_state[0], 2), round(self.discrete_state[1], 2), round(self.discrete_state[2], 2))
	key_sim = (round(self.discrete_state_sim[0], 2), round(self.discrete_state_sim[1], 2), round(self.discrete_state_sim[2], 2))
	print('STATE: ', key)
    if self.blind_spots[key] >= self.prob_thresh: # BLIND SPOT
        # print('STATE: ', key)
        print('BLIND SPOT: (0=forward, 1=left, 2=right)')
        # return input()
        return 0
    else: # NOT BLIND SPOT
        # print('STATE: ', key_sim)
        print('NOT BLIND SPOT')
        return self.sim_policy[key_sim]

if __name__ == "__main__":
    rospy.init_node('demo')
    demo = Demo()
    demo.generate_blind_spots() # dictionary of state:blind_spot_status
    demo.generate_sim_policy() # dictionary of state:action
    r = rospy.Rate(0.1)
    while not rospy.is_shutdown() and demo.run:
    	demo.set_state() # set discrete state
        a = demo.determine_action() # get the right action
    	print('ACTION: ', a)
    	t0 = rospy.get_rostime()
    	t1 = t0 + rospy.Duration(1.5)
        demo.pub_drive.publish(a) # publish drive command
        # demo.test_termination() # test if you've made it to the goal
    	r.sleep() 

    rospy.spin()
