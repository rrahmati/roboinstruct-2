#! /usr/bin/python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import sys
import os
from os.path import expanduser
import signal
import threading
from multiprocessing import Pool
import time
from random import randint
from std_msgs.msg import Float32MultiArray

from leap_client.msg import HandInfoList

def signal_handler(signal, frame):
    global record_demonstratio
    n
    record_demonstration.end_thread = True
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

class RecordDemonstration(object):

    def __init__(self):
        # parameters
        self.task = 3006
        # person controlling the robot: 1-Rouhollah, 2-Pooya
        self.user_id = 1
        self.image_shape = (540, 540)
        self.recordDelay = .03
        self.camera1 = True
        self.camera2 = False
        self.camera3 = False
        self.al5d = True
        self.mico = False
        self.task_description = {
            5000: "Human demonstrations",
            3001: "Grab a bubble wrap and put it into plate",
            3002: "Push the plate to the left",
            3003: "Push the box towards the robot's base",
            3004: "Push and roll the bottle towards the robot's base",
            3005: "Pick up the towel and clean the screwdriver box",
            3006: "rotate the pliers wrench to a perpendicular orientation",

            # first camera calibration:
            1001: "Put three small objects into the container",
            1002: "Grab a pen and put it into user's hand",
            1003: "Take the stirring bar from the user, stir a coffee cup, give it back to the user",
            1004: "Grab capsules from the table and put them into their bottle",
            1005: "Grab a paper cup and pour its content into a plate",
            1006: "Push all small cubes and gather them in the middle of table",
            1007: "The small towel is already folded. fold it one more time",
            1008: "Grab a paper cup and put it into a tea cup",
            1009: "Grab the spoon and fork and put them into the plate, spoon on right, fork on left",
            1010: "Pick up a thick marker and put it into upright position",
            1011: "Push and rotate the markers and gather them close to the robot base",
            1012: "Stay in the middle position. Don't move!",
            1013: "Pick up a mug and place it on the table where the user is pointing",
            1014: "scoop ...",
            # second camera calibration:
            1501: "Grab 6 small cubes in a cluttered situation and put them into a plate",
            1502: "Grab a marker and put it into the cup. Then, put it back on the table.",
            # second camera calibration, each task 5 minutes, 10,000 waypoints
            2001: "Grab 3 small markers and arrange them vertically on the right side",
            2002: "Grab 3 small markers and arrange them horizontally on the right side",
            2003: "Grab 3 small markers and arrange them vertically on the left side",
            2004: "Grab 3 small markers and arrange them horizontally on the left side",
            2005: "Grab 3 small markers and make a triangle with them",
            2006: "Grab 3 small markers, put one on the left, one on the right, and one in the middle",
            2007: "Grab 3 small markers and make a horizontal line with them",
            2008: "Grab 3 small markers and write the character Y with them",
            2009: "Grab 3 small markers and write the character U with them",
            2010: "Grab 3 small markers and write the character H with them",
            2011: "Grab 3 small markers and write the character N with them",
            2012: "Grab 3 small markers and write the character T with them",
            2013: "Grab 3 small markers and write the reversed character N with them",
            2014: "Grab 3 small markers and write the reversed character Y with them",
            2015: "Grab 3 small markers and write the reversed character U with them",
            2016: "Grab 3 small markers and write the 90 degree rotated character H with them",
            2017: "Grab 3 small markers and write the reversed character T with them",
            2018: "Grab 3 small markers and write the character K with them",
            2019: "Grab 3 small markers, put one vertically on the right, and two vertically on the left",
            2020: "Grab 3 small markers, put one vertically on the left, and two vertically on the right",
            2021: "Grab 3 small markers, put one horizontally on the right, and two horizontally on the left",
            2022: "Grab 3 small markers, put one horizontally on the left, and two horizontally on the right",
            2023: "Grab 3 small markers, put one vertically on the right, and two horizontally on the left",
            2024: "Grab 3 small markers, put one horizontally on the left, and two vertically on the right",
            2025: "Grab 3 small markers, put one vertically on the right, and make a vertical line with the other two",
            2026: "Grab 3 small markers, put one vertically on the left, and make a vertical line with the other two",
            2027: "Grab 3 small markers, put one vertically on the right, and make a horizontal line with the other two",
            2028: "Grab 3 small markers, put one vertically on the left, and make a horizontal line with the other two",
            2029: "Grab 3 small markers and put them into the coffee cup on the right",
            2030: "Grab 3 small markers that are inside a coffee cup on the right and put them on the desk",
            2031: "Grab 3 small markers and put them into the coffee cup on the left",
            2032: "Grab 3 small markers that are inside a coffee cup on the left and put them on the desk",
            2033: "Grab 3 small markers, put one into the coffee cup on the left, and the others into the coffee cup on the right",
            2034: "Grab 3 small markers, put one into the coffee cup on the right, and the others into the coffee cup on the left",
            2035: "Grab 2 small markers, put one into the coffee cup on the right, and the other into the coffee cup on the left",
            2036: "Grab 2 small markers, put one into the coffee cup on the left, and the other into the coffee cup on the right",
            2037: "Grab one small marker from each coffee cup and put them on the desk",
            2038: "Grab one small marker from the coffee cup on the right and put it into the coffee cup on the left",
            2039: "Grab one small marker from the coffee cup on the left and put it into the coffee cup on the right",
            2040: "Grab 4 small markers and make a square with them",
            2041: "Grab 4 small markers and make a cross with them",
            2042: "Grab 4 small markers and make a 45 degree rotated square with them",
            2043: "Grab 4 small markers and make a plus with them",
            2044: "Grab 4 small markers, put one vertically on the right and three vertically on the left",
            2045: "Grab 4 small markers, put one horizontally on the right and three vertically on the left",
            2046: "Grab 4 small markers, put one vertically on the right and three horizontally on the left",
            2047: "Grab 4 small markers, put one horizontally on the right and three horizontally on the left",
            2048: "Grab 4 small markers, put two vertically on the right and two vertically on the left",
            2049: "Grab 4 small markers, put two horizontally on the right and two vertically on the left",
            2050: "Grab 4 small markers, put two vertically on the right and two horizontally on the left",
            2051: "Grab 4 small markers, put two horizontally on the right and two horizontally on the left",
            2052: "Grab 4 small markers and draw the bottom half of a star with them",
            2053: "Grab 4 small markers and draw the upper half of a star with them",
            2054: "Grab 4 small markers and draw the character '=' with them",
            2055: "Grab 4 small markers and draw the 90 degree rotated character '=' with them",
            2056: "Grab 4 small markers and draw the character 'W' with them",
            2057: "Grab 4 small markers and draw the character 'M' with them",
            2058: "Grab 4 small markers and draw the character 'E' with them",
            2059: "Grab 4 small markers and draw the reversed character 'E' with them",
            2060: "Grab 4 small markers and draw the character 'm' with them",
            2061: "Grab 4 small markers and draw the reversed character 'm' with them",


            }
        # initialization
        self.filepath = expanduser("~") + '/t/task-' + str(self.task) + '/' + str(randint(0,1000000))

        rospy.init_node('record_demonstration')
        if self.camera1:
            self.create_folders(self.filepath + '/camera-' + str(1) + '/')
            # self.create_folders(self.filepath + '/camera-' + str(1) + '-depth/')
            rospy.Subscriber("/kinect2/qhd/image_color_rect", Image, self.camera1_callback)
            # rospy.Subscriber("/kinect2/hd/image_depth_rect", Image, self.camera1_depth_callback)
        if self.camera2:
            self.create_folders(self.filepath + '/camera-' + str(2) + '/')
            rospy.Subscriber("/usb_cam/image_raw", Image, self.camera2_callback)
        if self.camera3:
            self.create_folders(self.filepath + '/camera-' + str(3) + '/')
            rospy.Subscriber("/kinect2/qhd/image_color_rect", Image, self.camera3_callback)
        if self.al5d:
            self.write_file_header()
            rospy.Subscriber("/leap_al5d_info", Float32MultiArray, self.leap_al5d_callback)
        if self.mico:
            self.write_file_header()
            rospy.Subscriber("/leap_mico_info", Float32MultiArray, self.leap_mico_callback)
        self.bridge = CvBridge()
        self.timestep = 0
        self.task_complete_count = 0
        self.rate = rospy.Rate(self.recordDelay*1000)
        self.last_reward_time = 0
        self.last_robot_msg = 0
        self.start_time = rospy.get_time()
        self.end_thread = False
        self.pause = False
        # self.pool = Pool(2)
        self.thread = threading.Thread(target= self._update_thread)
        self.thread.start()

    def save_image(self, img_msg, camera):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            img = np.array(img, dtype=np.float)
        except CvBridgeError, e:
            print(e)
        else:
            img = img[0:540, 250:840]
            img = cv2.resize(img, self.image_shape)
            cv2.imwrite(self.filepath + '/camera-' + str(camera) + '/' + str(self.timestep) +
             '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    def save_image_depth(self, img_msg, camera):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            img = np.array(img, dtype=np.float32)
            cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        except CvBridgeError, e:
            print(e)
        else:
            img = cv2.resize(img, self.image_shape)
            cv2.imwrite(self.filepath + '/camera-' + str(camera) + '-depth/' + str(self.timestep) +
             '.jpg', img*255.0, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    def camera1_callback(self, msg):
        self.camera1_msg = msg

    def camera1_depth_callback(self, msg):
        self.camera1_depth_msg = msg

    def camera2_callback(self, msg):
        self.camera2_msg = msg

    def camera3_callback(self, msg):
        self.camera3_msg = msg

    def leap_al5d_callback(self, msg):
        self.leap_al5d_msg = msg
        self.last_robot_msg = rospy.get_time()

    def leap_mico_callback(self, msg):
        self.leap_mico_msg = msg

    def create_folders(self, foldername):
        if not os.path.exists(foldername):
            try:
                os.makedirs(foldername)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def write_file_header(self):
        with open(self.filepath + '.txt', 'w') as f:
            f.write(str(time.strftime('%l:%M%p %z on %b %d, %Y')) + '\n' + str(self.task_description[self.task]) + '\n')
            f.write('time,task,user,robot,reward,human,gripper,joint1,joint2,joint3,joint4,joint5,joint6')


    def append_to_file(self, robot):
        with open(self.filepath + '.txt', 'a') as f:
            str_to_append = '\n' + str(rospy.get_time() - self.start_time) + ',' + str(self.task) + ',' + str(self.user_id) + ','
            if robot == 'al5d':
                str_to_append = str_to_append + str(1) + ','
                data = [x for x in self.leap_al5d_msg.data]
            elif robot == 'mico':
                str_to_append = str_to_append + str(2) + ','
                data = [x for x in self.leap_mico_msg.data]

            if abs(data[0] - 1) < .01: # got reward
                if rospy.get_time() - self.last_reward_time > 1:
                    self.task_complete_count += 1
                    self.last_reward_time = rospy.get_time()
                else:
                    data[0] = 0
            sys.stdout.write('\rTimestep: ' + str(self.timestep) + ' Task done: ' + str(self.task_complete_count))
            sys.stdout.flush()

            str_to_append = str_to_append + ','.join(str(e) for e in data)
            f.write(str_to_append)

    def _update_thread(self):
        while not rospy.is_shutdown() and not self.end_thread:
            if self.pause or rospy.get_time() - self.start_time < 1 or rospy.get_time() - self.last_robot_msg > .1:
                continue
            save_files = (self.camera1 == hasattr(self, 'camera1_msg') and self.camera2 == hasattr(self, 'camera2_msg')
            and self.camera3 == hasattr(self, 'camera3_msg') and self.al5d == hasattr(self, 'leap_al5d_msg')
            and self.mico == hasattr(self, 'leap_mico_msg'))

            if save_files:
                if self.camera1:
                #     # self.pool.map(self.save_image, [(self.camera1_msg, 1)])
                    self.save_image(self.camera1_msg, 1)
                    # self.save_image_depth(self.camera1_depth_msg, 1)
                if self.camera2:
                    # self.pool.map(self.save_image, [(self.camera2_msg, 2)])
                    self.save_image(self.camera2_msg, 2)
                if self.camera3:
                    self.save_image(self.camera2_msg, 3)
                if self.al5d:
                    self.append_to_file('al5d')
                if self.mico:
                    self.append_to_file('mico')
                self.timestep += 1
            self.rate.sleep()
def main():

    global record_demonstration
    record_demonstration = RecordDemonstration()
    rospy.spin()
    # while not rospy.is_shutdown() and not record_demonstration.end_thread:
    #     input = raw_input(">>>")
    #     record_demonstration.pause = not record_demonstration.pause

if __name__ == '__main__':
    main()
