#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import math
import random
import threading
import numpy as np
from time import sleep
from scipy import ndimage
from collections import deque
import matplotlib.pyplot as plt
# from scipy import ndimage, signal
from skimage.transform import warp_polar
# from squaternion import Quaternion

import rclpy
import tf2_ros
import tf_transformations
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Empty as Emptym
from std_srvs.srv import Empty as Emptys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
# from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from rosgraph_msgs.msg import Clock as ClockMsg
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from tf2_geometry_msgs import PoseStamped as TF2PoseStamp
from cartographer_ros_msgs.srv import FinishTrajectory, StartTrajectory

class RLEnvTurtlebot(Node):
    def __init__(self):
        super().__init__('rl_env_turtlebot')
        # qos_profile
        qos_profile = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.launch_file_name = 'cartographer.launch.py'
        self.launch_package_name = 'turtlebot3_cartographer'
        self.launch_args = 'use_sim_time:=True'
        self.process = None

        self.start_launch_file()
        time.sleep(5)

        self.position = np.array((0,0,0))
        self.velocity = np.array((0,0,0))
        self.goal_position = np.array((0,0))
        self.polar_minimap = np.zeros([36,301])
        self.robot_radius = 0.15
        self.sim_time = None
        self.collision_detected = False
        self.real_time_factor = 1
        self.sleep_time = 0.5

        self.tf_restarting = False
        self.reset_time_publisher = self.create_publisher(Emptym, '/reset_time', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers that will trigger the callbacks
        self.create_subscription(Odometry, 'odom', self.odometry_callback, qos_profile)
        
        self.create_subscription(OccupancyGrid, 'map', self.map_callback, qos_profile)
        
        # Initialize a subscriber to the '/clock' topic
        self.create_subscription(ClockMsg, '/clock', self.clock_callback, qos_profile)

        # Initialize a subscriber to the LIDAR topic
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile)

        # Create the cmd_vel publisher
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Initialize tf listener in a separate thread to constantly update the robot's position.
        self.tf_listener_thread = threading.Thread(target=self.tf_listener_callback, daemon=True)
        self.tf_listener_thread.start()

        # Declare and get the goal tolerance parameter
        self.declare_parameter('goal_tolerance', 0.5)  # default to 0.5 meters
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value

        # Declare and get the max episode duration parameter
        self.declare_parameter('max_episode_duration', 300)  # default to 0.5 meters
        self.max_episode_duration = self.get_parameter('max_episode_duration').get_parameter_value().double_value

        self.finish_trajectory_client = self.create_client(FinishTrajectory, '/finish_trajectory')
        self.start_trajectory_client = self.create_client(StartTrajectory, '/start_trajectory')

        # Create a client for the reset_simulation service
        self.reset_simulation_client = self.create_client(Emptys, '/reset_simulation')
        # Wait for the service to be available
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_simulation service not available, waiting again...')

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        """
        
        # Compute the Euclidean distance
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def clock_callback(self, clock_msg):
        """
        Callback function to handle incoming clock messages from the '/clock' topic.
        """
        self.sim_time = clock_msg.clock.sec + clock_msg.clock.nanosec * 1e9

    def lidar_callback(self, msg):
        """
        Callback function to handle incoming LIDAR data.
        """
        # Check if any distance readings are less than the robot radius
        for distance in msg.ranges:
            if distance < self.robot_radius:
                self.collision_detected = True
                self.get_logger().info('Collision imminent or occurred!')
                break
        else:
            self.collision_detected = False

    def reset_tf_time(self):
            empty_msg = Emptym()
            self.reset_time_publisher.publish(empty_msg)
            self.get_logger().info("Published empty message to /reset_time to reset rostime")

    def tf_listener_callback(self):
        """
        Continuously listen for and process tf transforms.
        """
        while rclpy.ok():
            if not self.tf_restarting:
                try:
                    trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
                    # Update self.position with the new data
                    quaternion = (
                        trans.transform.rotation.x,
                        trans.transform.rotation.y,
                        trans.transform.rotation.z,
                        trans.transform.rotation.w
                    )
                    yaw = tf_transformations.euler_from_quaternion(quaternion)[2]
                    self.position = np.array([
                        trans.transform.translation.x,
                        trans.transform.translation.y,
                        yaw])

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    self.get_logger().error(f'Could not transform from map to base_footprint: {e}')
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def odometry_callback(self, msg: Odometry):
        """
        Callback function for processing odometry messages.
        """
        linear_x = msg.twist.twist.linear.x
        linear_y = msg.twist.twist.linear.y
        angular_z = msg.twist.twist.angular.z
        self.velocity = np.array([linear_x, linear_y, angular_z])

    def map_callback(self, msg):
        """
        Callback function for processing map messages and creating a polar minimap.
        """
        # Constants
        minimap_size = 201  # Size of the minimap
        half_size = minimap_size // 2  # Half of the minimap size
        crop_size = int(np.ceil(minimap_size * np.sqrt(2))) // 2  # Size to crop after zero-padding
    
        # Update the full map
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))
        data = np.where(data < 0, data / 2, data / 255)
        
        # Zero-pad the map
        padded_map = np.pad(data, pad_width=crop_size, mode='constant', constant_values=-0.5)
        # Find the robot's position in the rotated padded map
        map_x = int((self.position[0] - msg.info.origin.position.x) / msg.info.resolution) + crop_size
        map_y = int((self.position[1] - msg.info.origin.position.y) / msg.info.resolution) + crop_size

        # Crop the ROI around the robot position
        cropped_map = padded_map[map_y - crop_size : map_y + crop_size,
                                     map_x - crop_size: map_x + crop_size]
        # self.visualize_minimap(cropped_map)
        # Convert to polar coordinates
        polar_minimap = np.transpose(warp_polar(cropped_map,output_shape=(36,301), radius=half_size, scaling='linear', order=0))

        # Rotate the minimap based on the robot's current yaw (assumed to be in radians)
        polar_minimap = self.rotate_polar_image(polar_minimap, np.pi + self.position[2])
        
        ## Calculate the goal's position relative to the robot and convert it to polar coordinates
        self.polar_minimap = self.place_goal_polar_map(polar_minimap, self.goal_position, self.position, np.pi + self.position[2], msg.info.resolution * half_size)
        # self.visualize_minimap(self.polar_minimap)
    
    def place_goal_polar_map(self, polar_minimap, goal_position, robot_position, yaw, max_range):
        """
        Place the goal in the polar minimap.
        """
        # Convert the robot and goal positions into robot-centric coordinates
        goal_x_relative = goal_position[0] - robot_position[0]
        goal_y_relative = goal_position[1] - robot_position[1]

        # Calculate the goal's angle and distance relative to the robot
        goal_distance = np.sqrt(goal_x_relative**2 + goal_y_relative**2)
        goal_angle = np.arctan2(goal_y_relative, goal_x_relative) - yaw

        # Normalize the goal_angle to be within the range [0, 2*pi)
        goal_angle = (goal_angle + 2 * np.pi) % (2 * np.pi)

        # Find the polar coordinates of the goal in the minimap
        radial_index = int((goal_distance / max_range) * polar_minimap.shape[0])
        angular_index = int(goal_angle / (2 * np.pi) * polar_minimap.shape[1])

        # If the goal is within the polar minimap
        if radial_index < polar_minimap.shape[0]:
            # Place the goal in the minimap
            polar_minimap[radial_index, angular_index] = -1  # Assuming -2 denotes the goal
        else:
            # Find the edge pixel in the direction of the goal
            edge_index = polar_minimap.shape[0] - 1
            # Wrap the angular index if necessary
            angular_index %= polar_minimap.shape[1]
            # Set the edge pixel to denote the goal
            polar_minimap[edge_index, angular_index] = -1

        return polar_minimap

    def rotate_polar_image(self, polar_image, rotation_angle_radian):
        """
        Rotate a polar image by a specified angle.
    
        Parameters:
        - polar_image: A 2D numpy array representing the polar image.
        - rotation_angle_radian: The rotation angle in radian. Positive values rotate the image to the left.
    
        Returns:
        - A rotated polar image.
        """
        # The angular resolution (radian per pixel)
        angular_resolution = 2 * np.pi / polar_image.shape[1]
    
        # Calculate the number of pixels to shift
        pixel_shift = int(rotation_angle_radian / angular_resolution)
    
        # Rotate the image by rolling it horizontally
        rotated_polar_image = np.roll(polar_image, shift=-pixel_shift, axis=1)
    
        return rotated_polar_image

    def fixed_minimap_callback(self, msg):
        # Update the full map
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data)
        self.map_2d = np.reshape(data, (height, width))
        
        # Calculate the robot's position in map coordinates
        map_x = int((self.position[0] - msg.info.origin.position.x) / msg.info.resolution)
        map_y = int((self.position[1] - msg.info.origin.position.y) / msg.info.resolution)
        
        # Define the size of the minimap
        minimap_size = 101
        half_size = minimap_size // 2
        
        # Calculate the bounds of the minimap
        map_min_x = max(0, map_x - half_size)
        map_max_x = min(width, map_x + half_size + 1)
        map_min_y = max(0, map_y - half_size)
        map_max_y = min(height, map_y + half_size + 1)
        
        # Extract the minimap
        self.minimap = -1 * np.ones((minimap_size, minimap_size), dtype=int)

        # Determine the range of minimap indices that will be filled
        mini_min_x = half_size - min(map_x, half_size)
        mini_max_x = mini_min_x + (map_max_x - map_min_x)
        mini_min_y = half_size - min(map_y, half_size)
        mini_max_y = mini_min_y + (map_max_y - map_min_y)

        # Copy over the data from the main map to the minimap
        self.minimap[mini_min_y:mini_max_y, mini_min_x:mini_max_x] = self.map_2d[map_min_y:map_max_y, map_min_x:map_max_x]
        
        # Calculate the goal's position in map coordinates
        goal_map_x = int((self.goal_position[0] - msg.info.origin.position.x) / msg.info.resolution)
        goal_map_y = int((self.goal_position[1] - msg.info.origin.position.y) / msg.info.resolution)

        # Check if the goal is within the bounds of the map
        if map_min_x < goal_map_x < map_max_x and map_min_y < goal_map_y < map_max_y:
            # The goal is within the map bounds, set its value to -2
            self.minimap[goal_map_y - map_min_y + mini_min_y, goal_map_x - map_min_x + mini_min_x] = -2
        else:
            # The goal is outside the map bounds, find the closest edge in the minimap
            # Normalize the goal direction vector
            goal_dir_x = goal_map_x - map_x
            goal_dir_y = goal_map_y - map_y
            norm = np.sqrt(goal_dir_x ** 2 + goal_dir_y ** 2)
            goal_dir_x /= norm
            goal_dir_y /= norm

            # Find the edge pixel in the direction of the goal
            edge_x, edge_y = half_size, half_size  # Start from the center
            while 0 <= edge_x < minimap_size and 0 <= edge_y < minimap_size:
                edge_x += goal_dir_x
                edge_y += goal_dir_y

            # Convert float index to int and ensure it's within minimap bounds
            edge_x = np.clip(int(edge_x), 0, minimap_size - 1)
            edge_y = np.clip(int(edge_y), 0, minimap_size - 1)

            # Set the edge pixel to -2
            self.minimap[edge_y, edge_x] = -2
            
        # Apply the normalization rules to each element in the minimap
        self.minimap = np.where(self.minimap < 0, self.minimap / 2, self.minimap / 255)
        
        self.visualize_minimap(self.minimap)

    def visualize_minimap(self, minimap):
        """
        Visualize the polar minimap using matplotlib with a polar projection.
        """
        if not plt.isinteractive():
            plt.ion()
            plt.show()

        # Clear the existing figure if it exists, otherwise create a new one
        plt.figure("Polar Minimap").clf()

        fig = plt.figure("Polar Minimap", figsize=[5, 5])
        ax = fig.add_subplot(111, polar=True)

        # Assume minimap's dimensions are (num_radius_values, num_theta_values)
        num_radius_values, num_theta_values = minimap.shape
        theta = np.linspace(0, 2 * np.pi, num_theta_values)
        radius = np.linspace(0, 1, num_radius_values)
        
        # Create meshgrid for theta and radius
        thetas, radii = np.meshgrid(theta, radius)

        # Plot the minimap
        ax.pcolormesh(thetas, radii, minimap, shading='auto')
        plt.savefig('polar.png')
        plt.draw()
        plt.pause(0.05)

    def execute_action(self, linear_velocity, angular_velocity):
        """
        Publish the robot's velocity commands.
        """
        # Create a new Twist message
        cmd_vel_msg = Twist()

        # Set the linear and angular velocities
        cmd_vel_msg.linear.x = linear_velocity
        cmd_vel_msg.angular.z = angular_velocity

        # Publish the command
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        self.get_logger().info(f"Published cmd_vel: linear={linear_velocity}, angular={angular_velocity}")

    def calculate_dense_reward(self, robot_position, goal_position):
        """
        Calculate the dense reward based on the distance of the mobile robot from the goal.
        """
        distance_to_goal = self.calculate_distance(robot_position, goal_position)
        reward = -distance_to_goal  # Negative reward for the distance to the goal

        # Penalize the reward if a collision is detected
        if self.collision_detected:
            reward -= 10  # This is a large penalty for collision. Adjust as necessary.

        return reward
    
    def calculate_sparse_reward(self, robot_position, goal_position):
        """
        Calculate the sparse reward which is given only when the robot reaches the goal.
        """
        distance_to_goal = self.calculate_distance(robot_position, goal_position)
        reward = 0 if distance_to_goal <= self.goal_tolerance else -1
        return reward

    def is_done(self, robot_position=None, goal_position=None, sim_time=None):
        """
        Determine if the episode is done.
        """
        # Check if the goal is reached

        # Use the provided arguments if they are not None, otherwise use the instance attributes
        robot_position = robot_position if robot_position is not None else self.position[0:2]
        goal_position = goal_position if goal_position is not None else self.goal_position
        sim_time = sim_time if sim_time is not None else self.sim_time

        if self.calculate_distance(robot_position, goal_position) <= self.goal_tolerance:
            self.get_logger().info('Goal reached!')
            return True
        # Check if the maximum time for the episode has been exceeded
        if sim_time > self.max_episode_duration:
            self.get_logger().info('Maximum episode time exceeded.')
            return True
        
        return False

    def get_observation(self):
        """
        Retrieve the current observation of the environment.
        
        Returns:
        dict: A dictionary containing the 2D map, robot position, and robot velocity.
        """
        # Here we assume that the callbacks for pose, velocity, and map update these variables
        observation = {
            'map': self.polar_minimap,
            'pose_goal_velocity': np.concatenate([self.position, self.goal_position, self.velocity])
        }
        return observation

    def step(self, action):
        """
        Execute the given action in the environment and observe the result.

        Parameters:
        action (tuple): The action to take, typically a tuple containing linear and angular velocities.

        Returns:
        Tuple[np.ndarray, float, bool, dict]: A tuple containing the next state, reward, done flag, and additional info.
        """
        # Execute the action
        self.execute_action(*action)  # Unpack the action into its components

        # Assume some time passes in the simulation
        time.sleep(self.sleep_time / self.real_time_factor)

        # Get the next state
        next_state = self.get_observation()

        # Calculate the reward
        reward = self.calculate_dense_reward(self.position[0:2], self.goal_position)

        # Check if the episode is done
        done = self.is_done()

        # Collect any additional information
        info = {
            'is_collision': self.collision_detected,
            'sim_time': self.sim_time
        }

        return next_state, reward, done, info

    def reset(self):
        """
        Reset the Gazebo simulation.
        """

        self.stop_launch_file()
        # Call the reset_simulation service
        reset_future = self.reset_simulation_client.call_async(Emptys.Request())
        self.tf_restarting = True
        self.reset_tf_time()
        if reset_future.result() is not None:
            self.get_logger().info('Simulation has been reset')
        else:
            self.get_logger().error('Failed to call reset_simulation service')

        time.sleep(3)
        self.start_launch_file()
        time.sleep(5)
        self.reset_tf_time()
        self.tf_restarting = False
        # request = FinishTrajectory.Request()
        # request.trajectory_id = 0
        # finish_trajectory_future = self.finish_trajectory_client.call_async(request)
        # # rclpy.spin_until_future_complete(self, finish_trajectory_future)



        # request = StartTrajectory.Request()
        # request.configuration_directory = "/home/ali22/ros2_ws/install/turtlebot3_cartographer/share/turtlebot3_cartographer/config"
        # request.configuration_basename = "turtlebot3_lds_2d.lua"
        # request.use_initial_pose = False
        # request.initial_pose.position.x = 0.0
        # request.initial_pose.position.y = 0.0
        # request.initial_pose.position.z = 0.0
        # request.initial_pose.orientation.x = 0.0
        # request.initial_pose.orientation.y = 0.0
        # request.initial_pose.orientation.z = 0.0
        # request.initial_pose.orientation.w = 1.0
        # request.relative_to_trajectory_id = 0
        # start_trajectory_future = self.start_trajectory_client.call_async(request)
        # print(start_trajectory_future.result(),'########################')


        # if finish_trajectory_future.result() is not None:
        #     self.get_logger().info('Finish trajectory request sent.')
        # else:
        #     self.get_logger().error('Failed to call service /finish_trajectory')

        # if start_trajectory_future.result() is not None:
        #     self.get_logger().info('Start trajectory request sent.')
        # else:
        #     self.get_logger().error('Failed to call service /start_trajectory')

    # def reset(self):
    #     self.reset_world()
    #     self.joint1_command.publish(0)
    #     self.joint2_command.publish(0)
    #     self.joint3_command.publish(0)
    #     self.setPoses()

    #     sleep(0.1)
    #     self.prev_distance = np.sqrt((self.target_position[0]-self.object_position[0])**2+(
    #       self.target_position[1]-self.object_position[1])**2)
    #     self.table_hight = self.object_hight - 0.035

    #     obs = self.observation(True)

    #     return obs

    def start_launch_file(self):
        if self.process is not None:
            self.stop_launch_file()
        self.process = subprocess.Popen(['ros2', 'launch', self.launch_package_name, self.launch_file_name, self.launch_args])

    def stop_launch_file(self):
        if self.process is not None:
            self.process.send_signal(signal.SIGINT)  # Send SIGINT to stop the launch file
            self.process.wait()  # Wait for the process to terminate
            self.process = None

    def HER(self, state, next_state, reward, done, virtual_target):
        object_position = state[6:9]
        next_object_position = next_state[6:9]
        #target = state[0:2]
        virtual_state = np.concatenate((virtual_target[0:2], state[2:]))
        virtual_next_state = np.concatenate((virtual_target[0:2], next_state[2:]))

        #virtual_distance = np.sqrt(
        #    (virtual_target[0]-object_position[0])**2+(virtual_target[1]-object_position[1])**2)
        virtual_next_distance = np.sqrt(
            (virtual_target[0]-next_object_position[0])**2+(virtual_target[1]-next_object_position[1])**2)
        
        if virtual_next_distance < 0.02: virtual_reward = 0
        else: virtual_reward = -1
        
        if virtual_next_distance < 0.02 or done: virtual_done = 1
        else: virtual_done = 0

        return virtual_state, virtual_next_state, virtual_reward, virtual_done

# *** Check failure stack trace: ***
# google::LogMessage::Fail()
# google::LogMessage::SendToLog()
# google::LogMessage::Flush()
# google::LogMessageFatal::~LogMessageFatal()
# rclcpp::Executor::execute_service()
# rclcpp::Executor::execute_any_executable()
# rclcpp::executors::SingleThreadedExecutor::spin()
# rclcpp::spin()
# rclcpp::spin()
# __libc_start_main

# /start_trajectory
# cartographer_ros_msgs/srv/StartTrajectory

# configuration_directory
# /home/ali22/ros2_ws/install/turtlebot3_cartographer/share/turtlebot3_cartographer/config

# configuration_basename
# turtlebot3_lds_2d.lua

# use_initial_pose
# True

# initial_pose

# relative_to_trajectory_id
# 0
