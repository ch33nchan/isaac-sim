#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import threading
import numpy as np
import math
from math import sin, cos, atan2
from cv_bridge import CvBridge
import cv2
import itertools
import torch
import argparse
import datetime
#import time
import copy
import os
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from pxr import UsdGeom

bridge = CvBridge()

# enable ROS2 bridge extension
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

# spawn world
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)

path = os.environ["HOME"]
omni.usd.get_context().open_stage(path + "/Isaac_SAC/Isaac_SAC.usd")

# wait for things to load
simulation_app.update()
while omni.isaac.core.utils.stage.is_stage_loading():
    simulation_app.update()

simulation_context = omni.isaac.core.SimulationContext()
simulation_context.initialize_physics()
simulation_context.play()

body_pose = np.array([0.0, 0.0, 0.0], float) # x, y, theta
lidar_data = np.zeros(20)
image_L = 720
H = image_L
W = image_L
pix2m = 0.1 # m/pix
L = 8 # length of a box
stage_W = image_L*pix2m
stage_H = image_L*pix2m
clash_sum = 0

image = np.zeros((H, W, 3), np.uint8)
image_for_clash_calc = np.zeros((H, W), np.uint8)
# start region
pt27 = (int((stage_W/2 - L/2 + 24)/pix2m), int((stage_H/2 + L/2 + 24)/pix2m))
pt28 = (int((stage_W/2 + L/2 + 24)/pix2m), int((stage_H/2 - L/2 + 24)/pix2m))
cv2.rectangle(image, pt27, pt28, (0, 0, 255), cv2.FILLED, cv2.LINE_8)
# goal region
pt27 = (int((stage_W/2 - L/2 - 24)/pix2m), int((stage_H/2 + L/2 - 24)/pix2m))
pt28 = (int((stage_W/2 + L/2 - 24)/pix2m), int((stage_H/2 - L/2 - 24)/pix2m))
cv2.rectangle(image, pt27, pt28, (255, 0, 0), cv2.FILLED, cv2.LINE_8)

# for debug
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

class GazeboEnv(Node):

    def __init__(self):
        super().__init__('env')

        self.knuckle_pos = np.array([0,0], float)  #left right
        self.wheel_vel= np.array([0,0], float)     #left right
        self.publisher_pos = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.publisher_vel = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)

        self.T = 1.1  # track of the front and rear 
        self.L = 1.8 # wheel base
        self.Rw = 0.3 # Radius of the front and rear wheel

        self.next_state = np.zeros(22) # 0~19:lidara, 20:distance to the goal
        
        self.goal = [24.0, -24.0]
        
        self.vel = 3.0 # translation velocity of the car
        self.omega = 0.0 # angular velocity of the car

        self.cube_size = 8.0
        self.prev_dist = 0

        # Create a new transform prim (to hold our cylinder)
        self.cube_xform1 = XFormPrim(prim_path="/World/Cube1")
        self.cube_xform2 = XFormPrim(prim_path="/World/Cube2")
        self.cube_xform3 = XFormPrim(prim_path="/World/Cube3")
        self.cube_xform4 = XFormPrim(prim_path="/World/Cube4")
        self.cube_xform5 = XFormPrim(prim_path="/World/Cube5")
        self.cube_xform6 = XFormPrim(prim_path="/World/Cube6")
        self.cube_xform7 = XFormPrim(prim_path="/World/Cube7")
        self.cube_xform8 = XFormPrim(prim_path="/World/Cube8")
        self.cube_xform9 = XFormPrim(prim_path="/World/Cube9")
        self.cube_xform10 = XFormPrim(prim_path="/World/Cube10")
        self.cube_xform11 = XFormPrim(prim_path="/World/Cube11")
        self.cube_xform12 = XFormPrim(prim_path="/World/Cube12")
        self.cube_xform13 = XFormPrim(prim_path="/World/Cube13")
        self.cube_xform14 = XFormPrim(prim_path="/World/Cube14")

        # Create the cube geometry (using UsdGeom)
        stage = my_world.stage
        cube_geom1 = UsdGeom.Cube.Define(stage, "/World/Cube1/Geom")
        cube_geom2 = UsdGeom.Cube.Define(stage, "/World/Cube2/Geom")
        cube_geom3 = UsdGeom.Cube.Define(stage, "/World/Cube3/Geom")
        cube_geom4 = UsdGeom.Cube.Define(stage, "/World/Cube4/Geom")
        cube_geom5 = UsdGeom.Cube.Define(stage, "/World/Cube5/Geom")
        cube_geom6 = UsdGeom.Cube.Define(stage, "/World/Cube6/Geom")
        cube_geom7 = UsdGeom.Cube.Define(stage, "/World/Cube7/Geom")
        cube_geom8 = UsdGeom.Cube.Define(stage, "/World/Cube8/Geom")
        cube_geom9 = UsdGeom.Cube.Define(stage, "/World/Cube9/Geom")
        cube_geom10 = UsdGeom.Cube.Define(stage, "/World/Cube10/Geom")
        cube_geom11 = UsdGeom.Cube.Define(stage, "/World/Cube11/Geom")
        cube_geom12 = UsdGeom.Cube.Define(stage, "/World/Cube12/Geom")
        cube_geom13 = UsdGeom.Cube.Define(stage, "/World/Cube13/Geom")
        cube_geom14 = UsdGeom.Cube.Define(stage, "/World/Cube14/Geom")

        # Set cube attributes (optional)
        cube_geom1.GetSizeAttr().Set(self.cube_size)  # Set the size of the cube
        cube_geom2.GetSizeAttr().Set(self.cube_size)
        cube_geom3.GetSizeAttr().Set(self.cube_size)
        cube_geom4.GetSizeAttr().Set(self.cube_size)
        cube_geom5.GetSizeAttr().Set(self.cube_size)
        cube_geom6.GetSizeAttr().Set(self.cube_size)
        cube_geom7.GetSizeAttr().Set(self.cube_size)
        cube_geom8.GetSizeAttr().Set(self.cube_size)
        cube_geom9.GetSizeAttr().Set(self.cube_size)
        cube_geom10.GetSizeAttr().Set(self.cube_size)
        cube_geom11.GetSizeAttr().Set(self.cube_size)
        cube_geom12.GetSizeAttr().Set(self.cube_size)
        cube_geom13.GetSizeAttr().Set(self.cube_size)
        cube_geom14.GetSizeAttr().Set(self.cube_size)

    def step(self, action, time_steps, max_episode_steps):
        global body_pose, lidar_data, clash_sum

        self.done = False
        self.omega = 2*action[0]

        if((2*self.vel - self.omega*self.T) != 0):
            self.knuckle_pos[0] = math.atan(self.omega*self.L/(2*self.vel - self.omega*self.T))
        else:
            self.knuckle_pos[0] = 0
        
        if((2*self.vel + self.omega*self.T) != 0):
            self.knuckle_pos[1] = math.atan(self.omega*self.L/(2*self.vel + self.omega*self.T))
        else:
            self.knuckle_pos[1] = 0

        self.wheel_vel[0] = (self.vel - self.omega*self.T/2)/self.Rw
        self.wheel_vel[1] = (self.vel + self.omega*self.T/2)/self.Rw 

        #self.get_logger().info(f"self.wheel_vel:{self.wheel_vel}, knuckle_pos:{self.knuckle_pos}")

        wheel_vel_array = Float64MultiArray(data=self.wheel_vel)    
        self.publisher_vel.publish(wheel_vel_array)  
        knuckle_pos_array = Float64MultiArray(data=self.knuckle_pos)    
        self.publisher_pos.publish(knuckle_pos_array)  

        for _ in range(20):
            simulation_context.step(render=True)

        distance_to_the_goal = math.sqrt((body_pose[0]-self.goal[0])**2 + (body_pose[1]-self.goal[1])**2)

        self.next_state[:20] = lidar_data/30
        #self.next_state[20] = distance_to_the_goal/67.22 # 67.22 is initial distance from the goal
        self.next_state[20] = -(body_pose[0]-self.goal[0])/48 
        self.next_state[21] = (body_pose[1]-self.goal[1])/48

        # euler_angle[0]:rall, euler_angle[1]:pitch, euler_angle[2]:yaw
        if(clash_sum > 0):
           self.get_logger().info(f"CLASH. DONE.")
           self.done = True
           reward = -10
        elif(time_steps >= max_episode_steps):
           self.get_logger().info(f"TIME OUT. DONE.")
           self.done = True    
           reward = 0        
        elif( 20.0 < body_pose[0] < 28.0 and 
              -28.0 < body_pose[1] < -20.0):
            self.get_logger().info(f"GOAL. DONE.")
            self.done = True           
            reward = 20
        elif(min(lidar_data) < 3):
            self.get_logger().info(f"Too close to the wall.")
            reward = -5
        elif(distance_to_the_goal < self.prev_dist):
            reward = 10*max(1 - distance_to_the_goal/67.22, 0)
        else:
            reward = -1

        self.prev_dist = distance_to_the_goal

        self.get_logger().info(f"self.next_state:{self.next_state}, clash_sum:{clash_sum}, reward:{reward}, self.omega:{self.omega}")

        return self.next_state, reward, self.done

    def reset(self):
        global euler_angle

        simulation_context.reset()
        image_for_clash_calc[:,:] = 0
        image[:,:,:] = 0
        self.prev_dist = 0

        # start region
        pt27 = (int((stage_W/2 - L/2 + 24)/pix2m), int((stage_H/2 + L/2 + 24)/pix2m))
        pt28 = (int((stage_W/2 + L/2 + 24)/pix2m), int((stage_H/2 - L/2 + 24)/pix2m))
        cv2.rectangle(image, pt27, pt28, (0, 0, 255), cv2.FILLED, cv2.LINE_8)
        # goal region
        pt27 = (int((stage_W/2 - L/2 - 24)/pix2m), int((stage_H/2 + L/2 - 24)/pix2m))
        pt28 = (int((stage_W/2 + L/2 - 24)/pix2m), int((stage_H/2 - L/2 - 24)/pix2m))
        cv2.rectangle(image, pt27, pt28, (255, 0, 0), cv2.FILLED, cv2.LINE_8)

        rng = np.random.default_rng()
        boxes_pos = []
        for _ in range(7):
            numbers = rng.choice(9, size=2, replace=False)
            boxes_pos.append(numbers)

        for j in range(7):
            if j==0:
                if(boxes_pos[j][0] == 0):
                    coords = [-8, 32, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 1):
                    coords = [0.0, 32, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 2):
                    coords = [8, 32, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 3):
                    coords = [-8, 24, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 4):
                    coords = [0.0, 24, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 5):
                    coords = [8, 24, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 6):
                    coords = [-8, 16, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 7):
                    coords = [0.0, 16, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [8, 16, -2.3]
                    self.cube_xform1.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt1 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt2 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt1, pt2, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt1, pt2, 255, cv2.FILLED, cv2.LINE_8)

                if(boxes_pos[j][1] == 0):
                    coords = [-8, 32, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 1):
                    coords = [0.0, 32, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 2):
                    coords = [8, 32, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 3):
                    coords = [-8, 24, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 4):
                    coords = [0.0, 24, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 5):
                    coords = [8, 24, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 6):
                    coords = [-8, 16, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 7):
                    coords = [0.0, 16, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 8):
                    coords = [8, 16, -2.3]
                    self.cube_xform2.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt3 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt4 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt3, pt4, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt3, pt4, 255, cv2.FILLED, cv2.LINE_8)

            if j==1:
                print(f"boxes2:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    coords = [16, 32, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 1):
                    coords = [24, 32, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 2):
                    coords = [32, 32, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 3):
                    coords = [16, 24, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 4):
                    coords = [24, 24, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 5):
                    coords = [32, 24, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 6):
                    coords = [16, 16, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 7):
                    coords = [24, 16, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [32, 16, -2.3]
                    self.cube_xform3.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt5 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt6 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt5, pt6, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt5, pt6, 255, cv2.FILLED, cv2.LINE_8)

                if(boxes_pos[j][1] == 0):
                    coords = [16, 32, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 1):
                    coords = [24, 32, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 2):
                    coords = [32, 32, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 3):
                    coords = [16, 24, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 4):
                    coords = [24, 24, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 5):
                    coords = [32, 24, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 6):
                    coords = [16, 16, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 7):
                    coords = [24, 16, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 8):
                    coords = [32, 16, -2.3]
                    self.cube_xform4.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt7 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt8 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt7, pt8, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt7, pt8, 255, cv2.FILLED, cv2.LINE_8)

            if j==2:
                print(f"boxes3:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    coords = [-32, 8, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 1):
                    coords = [-24, 8, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 2):
                    coords = [-16, 8, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 3):
                    coords = [-32, 0.0, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 4):
                    coords = [-24, 0.0, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 5):
                    coords = [-16, 0.0, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 6):
                    coords = [-32, -8, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 7):
                    coords = [-24, -8, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [-16, -8, -2.3]
                    self.cube_xform5.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt9 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt10 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt9, pt10, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt9, pt10, 255, cv2.FILLED, cv2.LINE_8)

                if(boxes_pos[j][1] == 0):
                    coords = [-32, 8, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 1):
                    coords = [-24, 8, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 2):
                    coords = [-16, 8, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 3):
                    coords = [-32, 0.0, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 4):
                    coords = [-24, 0.0, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 5):
                    coords = [-16, 0.0, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 6):
                    coords = [-32, -8, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 7):
                    coords = [-24, -8, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 8):
                    coords = [-16, -8, -2.3]
                    self.cube_xform6.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt11 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt12 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt11, pt12, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt11, pt12, 255, cv2.FILLED, cv2.LINE_8)

            if j==3:
                print(f"boxes4:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    coords = [-8, 8, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 1):
                    coords = [0.0, 8, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 2):
                    coords = [8, 8, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 3):
                    coords = [-8, 0.0, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 4):
                    coords = [0.0, 0.0, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 5):
                    coords = [8, 0.0, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 6):
                    coords = [-8, -8, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 7):
                    coords = [0.0, -8, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [8, -8, -2.3]
                    self.cube_xform7.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt13 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt14 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt13, pt14, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt13, pt14, 255, cv2.FILLED, cv2.LINE_8)

                if(boxes_pos[j][1] == 0):
                    coords = [-8, 8, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 1):
                    coords = [0.0, 8, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 2):
                    coords = [8, 8, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 3):
                    coords = [-8, 0.0, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 4):
                    coords = [0.0, 0.0, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 5):
                    coords = [8, 0.0, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 6):
                    coords = [-8, -8, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 7):
                    coords = [0.0, -8, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 8):
                    coords = [8, -8, -2.3]
                    self.cube_xform8.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt15 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt16 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt15, pt16, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt15, pt16, 255, cv2.FILLED, cv2.LINE_8)

            if j==4:
                print(f"boxes5:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    coords = [32, 8, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 1):
                    coords = [24, 8, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 2):
                    coords = [16, 8, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 3):
                    coords = [32, 0.0, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 4):
                    coords = [24, 0.0, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 5):
                    coords = [16, 0.0, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 6):
                    coords = [32, -8, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 7):
                    coords = [24, -8, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [16, -8, -2.3]
                    self.cube_xform9.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt17 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt18 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt17, pt18, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt17, pt18, 255, cv2.FILLED, cv2.LINE_8)

                if(boxes_pos[j][1] == 0):
                    coords = [32, 8, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 1):
                    coords = [24, 8, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 2):
                    coords = [16, 8, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 3):
                    coords = [32, 0.0, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 4):
                    coords = [24, 0.0, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 5):
                    coords = [16, 0.0, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 6):
                    coords = [32, -8, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 7):
                    coords = [24, -8, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 8):
                    coords = [16, -8, -2.3]
                    self.cube_xform10.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt19 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt20 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt19, pt20, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt19, pt20, 255, cv2.FILLED, cv2.LINE_8)

            if j==5:
                print(f"boxes6:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    coords = [-32, -16, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 1):
                    coords = [-24, -16, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 2):
                    coords = [-16, -16, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 3):
                    coords = [-32, -24, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 4):
                    coords = [-24, -24, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 5):
                    coords = [-16, -24, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 6):
                    coords = [-32, -32, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 7):
                    coords = [-24, -32, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [-16, -32, -2.3]
                    self.cube_xform11.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt21 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt22 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt21, pt22, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt21, pt22, 255, cv2.FILLED, cv2.LINE_8)

                if(boxes_pos[j][1] == 0):
                    coords = [-32, -16, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 1):
                    coords = [-24, -16, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 2):
                    coords = [-16, -16, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 3):
                    coords = [-32, -24, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 4):
                    coords = [-24, -24, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 5):
                    coords = [-16, -24, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 6):
                    coords = [-32, -32, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 7):
                    coords = [-24, -32, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [-16, -32, -2.3]
                    self.cube_xform12.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt23 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt24 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt23, pt24, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt23, pt24, 255, cv2.FILLED, cv2.LINE_8)

            if j==6:
                print(f"boxes7:{boxes_pos[j][0]}, {boxes_pos[j][1]}")
                if(boxes_pos[j][0] == 0):
                    coords = [-8, -16, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 1):
                    coords = [0.0, -16, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 2):
                    coords = [8, -16, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 3):
                    coords = [-8, -24, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 4):
                    coords = [0.0, -24, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 5):
                    coords = [8, -24, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 6):
                    coords = [-8, -32, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 7):
                    coords = [0.0, -32, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [8, -32, -2.3]
                    self.cube_xform13.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt25 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt26 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt25, pt26, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt25, pt26, 255, cv2.FILLED, cv2.LINE_8)

                if(boxes_pos[j][1] == 0):
                    coords = [-8, -16, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 1):
                    coords = [0.0, -16, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 2):
                    coords = [8, -16, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 3):
                    coords = [-8, -24, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 4):
                    coords = [0.0, -24, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 5):
                    coords = [8, -24, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][1] == 6):
                    coords = [-8, -32, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 7):
                    coords = [0.0, -32, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                elif(boxes_pos[j][0] == 8):
                    coords = [8, -32, -2.3]
                    self.cube_xform14.set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                pt27 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt28 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt27, pt28, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt27, pt28, 255, cv2.FILLED, cv2.LINE_8)

        image_for_clash_calc[0:4, :] = 255
        image_for_clash_calc[716:720, :] = 255
        image_for_clash_calc[:, 0:4] = 255
        image_for_clash_calc[:, 716:720] = 255

        self.done = False
        self.next_state[:] = 0.0

        return self.next_state

class Get_modelstate(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, data):
        global body_pose

        pose = data.transforms[1].transform.translation
        orientation = data.transforms[1].transform.rotation

        body_pose[0] = pose.x
        body_pose[1] = pose.y
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w
        body_pose[2] = -atan2(2*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2))

class Lidar_subscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/laser_scan',
            self.listener_callback,
            10)
        self.subscription

        self.lidar_data_prev_step = np.zeros(20)

    def listener_callback(self, data):
        global lidar_data

        for i in range(20):
            value = data.ranges[180*i:180*i + 8]
            lidar_data[i] = np.max(value)
            if(lidar_data[i] <= 0):
                lidar_data[i] = self.lidar_data_prev_step[i]
            self.lidar_data_prev_step = copy.copy(lidar_data)

class Clash_calculation(Node):

    def __init__(self, image_L, pix2m):
        super().__init__('clash_calculation')
        self.time_interval = 0.05

        self.image_L = image_L
        self.H = self.image_L
        self.W = self.image_L
        self.C_H = int(self.H/2) 
        self.C_W = int(self.W/2)
        self.pix2m = pix2m
        self.robot_L = int(2.8/self.pix2m) 
        self.robot_W = int(1.3/self.pix2m) 
  
        margin = int(0.4/self.pix2m)
        self.OBL = self.robot_L + 2*margin # outer_boundary_Length
        self.OBW = self.robot_W + 2*margin # outer_boundary_Width
        
        self.robot1_region = np.zeros((self.H, self.W), np.uint8)

        self.pub_sum_img = self.create_publisher(Image, '/sum_image', 10)
        self.pub_common_part_img = self.create_publisher(Image, '/common_part_image', 10)
        self.timer = self.create_timer(self.time_interval, self.timer_callback)

    def timer_callback(self):
        global body_pose, clash_sum

        self.image = copy.copy(image)
        self.image_for_clash_calc = copy.copy(image_for_clash_calc)
        self.robot1_region[:,:] = 0

        theta1 = body_pose[2]
        self.robot1_ob = [[int(cos(theta1)*self.OBL/2 - sin(theta1)*self.OBW/2 + self.C_W + body_pose[0]/self.pix2m),       int(sin(theta1)*self.OBL/2 + cos(theta1)*self.OBW/2 + self.C_H - body_pose[1]/self.pix2m)],
                          [int(cos(theta1)*self.OBL/2 - sin(theta1)*(-self.OBW/2) + self.C_W + body_pose[0]/self.pix2m),    int(sin(theta1)*self.OBL/2 + cos(theta1)*(-self.OBW/2) + self.C_H - body_pose[1]/self.pix2m)],
                          [int(cos(theta1)*(-self.OBL/2) - sin(theta1)*(-self.OBW/2) + self.C_W + body_pose[0]/self.pix2m), int(sin(theta1)*(-self.OBL/2) + cos(theta1)*(-self.OBW/2) + self.C_H - body_pose[1]/self.pix2m)],
                          [int(cos(theta1)*(-self.OBL/2) - sin(theta1)*self.OBW/2 + self.C_W + body_pose[0]/self.pix2m),    int(sin(theta1)*(-self.OBL/2) + cos(theta1)*self.OBW/2 + self.C_H - body_pose[1]/self.pix2m)]]        
        pts1_ob = np.array(self.robot1_ob, np.int32)
        cv2.fillPoly(self.image, [pts1_ob], (80, 80, 80))
        cv2.fillPoly(self.robot1_region, [pts1_ob], 255)

        self.robot1_coord = [[int(cos(theta1)*self.robot_L/2 - sin(theta1)*self.robot_W/2 + self.C_W + body_pose[0]/self.pix2m),       int(sin(theta1)*self.robot_L/2 + cos(theta1)*self.robot_W/2 + self.C_H - body_pose[1]/self.pix2m)],
                             [int(cos(theta1)*self.robot_L/2 - sin(theta1)*(-self.robot_W/2) + self.C_W + body_pose[0]/self.pix2m),    int(sin(theta1)*self.robot_L/2 + cos(theta1)*(-self.robot_W/2) + self.C_H - body_pose[1]/self.pix2m)],
                             [int(cos(theta1)*(-self.robot_L/2) - sin(theta1)*(-self.robot_W/2) + self.C_W + body_pose[0]/self.pix2m), int(sin(theta1)*(-self.robot_L/2) + cos(theta1)*(-self.robot_W/2) + self.C_H - body_pose[1]/self.pix2m)],
                             [int(cos(theta1)*(-self.robot_L/2) - sin(theta1)*self.robot_W/2 + self.C_W + body_pose[0]/self.pix2m),    int(sin(theta1)*(-self.robot_L/2) + cos(theta1)*self.robot_W/2 + self.C_H - body_pose[1]/self.pix2m)]]
        pts1 = np.array(self.robot1_coord, np.int32)

        cv2.fillPoly(self.image, [pts1], (255, 165, 0))
        cv2.line(self.image, (int(cos(theta1)*self.robot_L/2 - sin(theta1)*self.robot_W/2 + self.C_W + body_pose[0]/self.pix2m), int(sin(theta1)*self.robot_L/2 + cos(theta1)*self.robot_W/2 + self.C_H - body_pose[1]/self.pix2m)),
                             (int(cos(theta1)*self.robot_L/2 - sin(theta1)*(-self.robot_W/2) + self.C_W + body_pose[0]/self.pix2m), int(sin(theta1)*self.robot_L/2 + cos(theta1)*(-self.robot_W/2) + self.C_H - body_pose[1]/self.pix2m)),
                              color=(0, 0, 255), thickness=3, lineType=cv2.LINE_4, shift=0)
        
        result_img = cv2.bitwise_and(self.robot1_region, self.image_for_clash_calc)
        clash_sum = cv2.countNonZero(result_img)
 
        img_common_part = bridge.cv2_to_imgmsg(result_img)  
        self.pub_common_part_img.publish(img_common_part)

        img_msg = bridge.cv2_to_imgmsg(self.image) 
        self.pub_sum_img.publish(img_msg) 
    

if __name__ == '__main__':
    rclpy.init(args=None)

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="obstacle_avoidance",
                    help='quadruped_isaac')
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=4000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=500000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
    args = parser.parse_args()

    env = GazeboEnv()
    get_modelstate = Get_modelstate()
    lidar_subscriber = Lidar_subscriber()
    clash_calculation = Clash_calculation(image_L, pix2m)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(env)
    executor.add_node(get_modelstate)
    executor.add_node(lidar_subscriber)
    executor.add_node(clash_calculation)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = env.create_rate(2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    action_space = np.zeros(1) # velocity, and steering_angle
    agent = SAC(len(env.next_state), action_space, args)

    agent.load_checkpoint(os.environ['HOME'] + "/Isaac_SAC/src/robot_control/scripts/checkpoints/sac_checkpoint_obstacle_avoidance_440", evaluate=True)

    max_episode_steps = 100
    
    try:
        while rclpy.ok():
            avg_reward = 0.
            episodes = 10
            for i  in range(episodes):
                print(f"eval episode{i}")
                state = env.reset()
                episode_reward = 0
                eval_steps = 0
                done = False
                while not done:
                    #print(f"step:{eval_steps}/{max_episode_steps}")
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done = env.step(action, eval_steps, max_episode_steps)
                    episode_reward += reward

                    eval_steps += 1
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
