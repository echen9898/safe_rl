# Standard imports
import time
import numpy as np
from math import sin, cos, pi, atan, degrees, radians
from itertools import product

# Gym imports
import gym
from gym import utils, spaces
from gazebo_env import GazeboEnv
from gym.utils import seeding

# ROS Standard imports
import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# ROS Message imports
from std_msgs.msg import Int8, Float32
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Quaternion, PolygonStamped, PoseWithCovarianceStamped, PoseWithCovariance, PointStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry


class LearningEnv(gym.Env):

    def __init__(self):

        # Environment
        self.vision_field_limit = radians(70)
        self.action_space = spaces.Discrete(3)
        self.allowed_values = [[_ for _ in np.arange(0.0, 3.0, 0.4)], [_ for _ in np.arange(-1.19, 1.19, 0.48)], [_ for _ in np.arange(-self.vision_field_limit, self.vision_field_limit, 0.45)]]
        self.feature_bins = [[-1000] + [_ for _ in np.arange(0.0, 3.0, 0.4)], [-1000] + [_ for _ in np.arange(-1.19, 1.19, 0.48)], [-1000] + [_ for _ in np.arange(-self.vision_field_limit, self.vision_field_limit, 0.45)]]
        self.dx = 0.249
        self.dy = 0.198
        self.left = radians(135) # far angle (160, 135)
        self.right = radians(45) # near angle (20, 45)
        self.num_angles = 8
        self.allowed_values = [[_ for _ in np.arange(0.0, 3.0, self.dx)], [_ for _ in np.arange(-1.19, 1.19, self.dy)], [_ for _ in np.arange(-pi, pi, (2*pi)/self.num_angles)]] 
        self.feature_bins = [[_ for _ in np.arange(0.0, 3.0, self.dx)], [_ for _ in np.arange(-1.19, 1.19, self.dy)], [_ for _ in np.arange(-pi, pi, (2*pi)/self.num_angles)]]
        self.states = list()

        # Pub/Sub
        self.pose_listener    = tf.TransformListener()
        self.pub_initial_pose = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.pub_drive        = rospy.Publisher('/drive_commands', Int8, queue_size=10) # publish drive commands

        # Robot State
        self.goal = np.array([2.86, 0.0])
        self.location = np.array([0.0, 0.0, 0.0]) # [x, y, heading] 
        self.state = np.array([2.86, 0.0, 0.0]) # [x_distance, y_distance, heading_distance]

        # Trackers
        self.terminate = False
        self.timestep = 0

        # Actions
        self.headings = np.array([0.0, -self.right, -radians(90), -self.left, -radians(180), self.left, radians(90), self.right])
        self.actions = { # heading : [forward, left, right] -> [[dx, dy, new_heading], [dx, dy, new_heading], [dx, dy, new_heading]]
        0.0 : [[self.dx, 0.0, 0.0], [self.dx, self.dy, self.right], [self.dx, -self.dy, -self.right]],
        -self.right : [[self.dx, -self.dy, -self.right], [self.dx, 0.0, 0.0], [0.0, -self.dy, -radians(90)]],
        -radians(90) : [[0.0, -self.dy, -radians(90)], [self.dx, -self.dy, -self.right], [-self.dx, -self.dy, -self.left]],
        -self.left : [[-self.dx, -self.dy, -self.left], [0.0, -self.dy, -radians(90)], [-self.dx, 0.0, -radians(180)]],
        -radians(180) : [[-self.dx, 0.0, -radians(180)], [-self.dx, -self.dy, -self.left], [-self.dx, self.dy, self.left]],
        self.left : [[-self.dx, self.dy, self.left], [-self.dx, 0.0, -radians(180)], [0.0, self.dy, radians(90)]],
        radians(90) : [[0.0, self.dy, radians(90)], [-self.dx, self.dy, self.left], [self.dx, self.dy, self.right]],
        self.right : [[self.dx, self.dy, self.right], [0.0, self.dy, radians(90)], [self.dx, 0.0, 0.0]]
        }


    def get_states(self):
        for s in product(*self.allowed_values):
            self.states.append(s)
        self.states.append([-1000, -1000, -1000])
        return self.states


    def assign_reward(self, x, y, h, dx, dy, dheading):
        reward = 0 
        # if (2.47 <= x <= 2.9) and (-0.37 <= y <= 0.37):
        if (0 <= dx <= 0.85) and (-0.45 <= dy <= 0.45):
            if y < 0 and (radians(20) <= h <= radians(70)): # right
                reward += 100 # reaching the goal is highly rewarded
            elif y > 0 and (-radians(70) <= h <= -radians(20)): # left
                reward += 100
            elif y == 0 and (-radians(10) <= h <= radians(10)): # center
                reward += 100
        else:
            reward -= 1
            # reward += ((3.0 - dx) + (1.19 - abs(dy)) + (self.vision_field_limit - abs(dheading)))
        return reward


    def termination_test(self, x, y, h, dx, dy, dheading):

        # Out of bounds
        if not (-0.105 < x < 2.9): # forward/backward check
            self.terminate = True
        if not (-1.19 < y < 1.19): # left/right check
            self.terminate = True

        # Reached the goal
        # if (2.05 <= x <= 2.9) and (-0.435 <= y <= 0.435):
        if (0 <= dx <= 0.85) and (-0.45 <= dy <= 0.45):
            if y < 0 and (radians(20) <= h <= radians(70)): # right
                self.terminate = True
            elif y > 0 and (-radians(70) <= h <= -radians(20)): # left
                self.terminate = True
            elif y == 0 and (-radians(10) <= h <= radians(10)): # center
                self.terminate = True

        # Timed out
        if self.timestep == 100:
            self.terminate = True


    def get_state(self, x, y, heading):

        # Determine heading coordinates (centered at robot location)
        right = radians(90)
        straight = radians(180)
        if -right < heading <= 0: # top right
            temp = right + heading
        elif -straight <= heading <= -right: # bottom right
            temp = heading + right
        elif 0 < heading <= right: # top left
            temp = right + heading
        elif right < heading <= straight: # bottom left
            temp = - right - straight + heading 
        yh = -cos(temp) # assume length 1 heading vector
        xh = sin(temp)

        # Determine rotated heading coordinates, and change in heading required
        dx = self.goal[0] - x # top side = positive, bottom side = positive
        dy = self.goal[1] - y # right side = positive, left side = negative

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


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _step(self, action):

        original_state = self.state[::1]

        # Take action
        self.timestep += 1
        x_old, y_old, heading_old = self.location # current location

        # Correct floating point error (very crudely lol)
        if -0.15 <= heading_old <= 0.15:
            heading_old = 0.0
        elif -self.right - 0.15 <= heading_old <= -self.right + 0.15:
            heading_old = -self.right
        elif -radians(90) - 0.15 <= heading_old <= -radians(90) + 0.15:
            heading_old = -radians(90)
        elif -self.left - 0.15 <= heading_old <= -self.left + 0.15:
            heading_old = -self.left
        elif -radians(180) - 0.15 <= heading_old <= -radians(180) + 0.15:
            heading_old = -radians(180)
        elif self.left - 0.15 <= heading_old <= self.left + 0.15:
            heading_old = self.left
        elif radians(90) - 0.15 <= heading_old <= radians(90) + 0.15:
            heading_old = radians(90)
        elif self.right - 0.15 <= heading_old <= self.right + 0.15:
            heading_old = self.right

        # Snap to new grid location
        dx_new, dy_new, heading_new = self.actions[heading_old][action]
        x_new = x_old + dx_new
        y_new = y_old + dy_new

        # Publish new pose
        initial_pose = PoseWithCovarianceStamped()
        pose = Pose()
        pose.position.x = x_new
        pose.position.y = y_new
        pose.position.z = 0.0
        random_quaterion = quaternion_from_euler(0.0, 0.0, heading_new) # arguments: roll, pitch, yaw
        pose.orientation.x = random_quaterion[0]
        pose.orientation.y = random_quaterion[1]
        pose.orientation.z = random_quaterion[2]
        pose.orientation.w = random_quaterion[3]
        initial_pose.pose.pose = pose
        self.pub_initial_pose.publish(initial_pose)
        rospy.sleep(0.05)

        # Update position and heading
        rotation = self.pose_listener.lookupTransform('/map', '/base_link', rospy.Time(0))[1]
        x, y = self.pose_listener.lookupTransform('/map', '/base_link', rospy.Time(0))[0][:2]
        heading = euler_from_quaternion(rotation)[2]
        dx, dy, dheading = self.get_state(x, y, heading)

        # Update state
        self.location = np.array([x, y, heading])
        if -self.vision_field_limit <= dheading <= self.vision_field_limit:
            self.state = np.array([dx, dy, dheading]) # goal in field of vision
        else:
            self.state = np.array([-1000, -1000, -1000]) # goal not seen

        # Compute reward
        reward = self.assign_reward(x, y, heading, dx, dy, dheading)

        # Test for termination
        self.termination_test(x, y, heading, dx, dy, dheading)

        # Return values (observation, reward, done, info)
        print('GOAL: ', self.goal)
        print('LOCATION: ', self.location[:2])
        print('STATE: ', self.state[:2])
        print('DHEADING: ', degrees(self.state[2]))
        print('REWARD: ', reward)
        print('TERMINATE: ', self.terminate)
        print('\n')
        return self.state, reward, self.terminate, None


    def _reset(self):

        # INITIAL PARAMETERS
        self.timestep = 0
        self.terminate = False

        # INITIAL POSITION/STATE

        # get a random seed
        x = self.goal[0] - np.random.choice(self.allowed_values[0]) # skip first two, in order to not start in goal region
        y = self.goal[1] - np.random.choice(self.allowed_values[1])
        heading = np.random.choice(self.headings)
        self.location = np.array([x, y, heading])
        dx, dy, dheading = self.get_state(x, y, heading)
        self.state = np.array([dx, dy, dheading])

        # publish an initial pose
        initial_pose = PoseWithCovarianceStamped()
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        random_quaterion = quaternion_from_euler(0.0, 0.0, heading) # arguments: roll, pitch, yaw
        pose.orientation.x = random_quaterion[0]
        pose.orientation.y = random_quaterion[1]
        pose.orientation.z = random_quaterion[2]
        pose.orientation.w = random_quaterion[3]
        initial_pose.pose.pose = pose
        self.pub_initial_pose.publish(initial_pose)

        return self.state