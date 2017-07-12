#! /usr/bin/python

import sys
import os
import time
import rospy
from tf.transformations import *
import signal
import copy
import numpy as np
from std_msgs.msg import Float32MultiArray
from ctypes import *
moveapi = cdll.LoadLibrary("/home/rouhollah/development/psmoveapi/build/_psmove.so")
trackerapi = cdll.LoadLibrary("/home/rouhollah/development/psmoveapi/build/libpsmoveapi_tracker.so")
# _psmove = cdll.LoadLibrary("/home/rouhollah/development/psmoveapi/build/_psmove.so")
moveapi.psmove_init(0x030901)

def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

print moveapi.psmove_count_connected()

tracker = trackerapi.psmove_tracker_new()
fusion = moveapi.psmove_fusion_new(tracker, 1, 1000)
trackerapi.psmove_tracker_set_mirror(tracker, 1);
trackerapi.psmove_tracker_set_exposure(tracker, 2);
move = moveapi.psmove_connect()
trackerapi.psmove_enable_orientation(move, 1)
while trackerapi.psmove_tracker_enable(tracker, move) != 2:
    pass
counter = 0
rospy.init_node('psmove')

test_mode = False
gripper = 1
last_button_time = 0
human = 0
human_mode = False
pause = True
position = np.zeros((3,))
orientation = np.zeros((4,))
saved_position = np.zeros((3,))
saved_orientation = np.zeros((4,))
move_msg = Float32MultiArray()
pause_counter = 0
move_pub = rospy.Publisher("/move_info_for_test" if test_mode else "/move_info", Float32MultiArray, queue_size=100)
while True:
    reward = 0
    while moveapi.psmove_poll(move):
        pass
    moveapi.psmove_tracker_update_image(tracker);
    moveapi.psmove_tracker_update(tracker, 0);
    moveapi.psmove_tracker_annotate(tracker);
    trackerapi.psmove_tracker_update_image(tracker);
    trackerapi.psmove_tracker_update(tracker, 0);
    x = c_float()
    y = c_float()
    z = c_float()
    w = c_float()
    moveapi.psmove_get_orientation(move, byref(x), byref(y), byref(z), byref(w))
    orientation = np.array([x.value, y.value, z.value, w.value])
    moveapi.psmove_tracker_get_position(tracker, move, byref(x), byref(y), byref(z))
    position = np.array([x.value/15., y.value/15., z.value*2.])
    if not moveapi.psmove_has_orientation(move):
        system.exit(0)
    button = moveapi.psmove_get_buttons(move)
    if rospy.get_time() - last_button_time > .5:
        if button == 128:

            last_button_time = rospy.get_time()
        if button == 64:
            saved_position = position - saved_position
            saved_orientation = orientation - saved_orientation
            pause = not pause
            pause_counter += 1
            if human_mode:
                if pause:
                    move_msg.data = [-7.000000000000002, -3.333333333333332, 0.45013427734375, 0.03623247146606445, -0.06508475542068481, -0.024073563516139984, 0.06926485151052475, 0.0, -13.5040283203125, 69.99999999999994, 330.0, -0.10000000000000001, 0.0, 1, 0, human]
                    move_pub.publish(move_msg)
                    print 'pause ' + str(pause_counter/4)
                else:
                    human = 1 - human
                    print 'human' if human == 1 else 'robot'
                    print 'recording...'
            last_button_time = rospy.get_time()
        if button == 32:
            gripper = 1 - gripper
            last_button_time = rospy.get_time()
        if button == 65536:
            sys.exit(0)
        if button == 16:
            reward = 1
            last_button_time = rospy.get_time()
    counter += 1
    position -= saved_position
    orientation -= saved_orientation
    if pause:
        time.sleep(.01)
        continue

    speed = 3.
    robot_position = (position ) * 10 * speed
    robot_position = np.array([-robot_position[2], robot_position[0], -robot_position[1]])
    robot_position[0] += 0
    robot_position[1] += 280
    robot_position[2] += 230

    ori_euler = np.array(euler_from_quaternion(numpy.array(orientation)))
    robot_ori = (ori_euler + 3.14 )
    for i in range(0, len(robot_ori)):
        if robot_ori[i] < 0:
            robot_ori[i] += 2 * 3.14
    robot_ori[0:] -= 3.14
    robot_ori *= 1
    robot_ori[0] = - robot_ori[0]
    # print robot_position, robot_ori
    robot_ori[0] = -.1
    robot_ori[1] = 0
    if human == 1:
        move_msg.data = [-7.000000000000002, -3.333333333333332, 0.45013427734375, 0.03623247146606445, -0.06508475542068481, -0.024073563516139984, 0.06926485151052475, 0.0, -13.5040283203125, 69.99999999999994, 330.0, -0.10000000000000001, 0.0, 1, 0, human]
    else:
        move_msg.data = position.tolist() + orientation.tolist() + [float(button)] + robot_position.tolist() + [robot_ori[0], robot_ori[1] , gripper, reward, human]
    move_pub.publish(move_msg)
    # print move_msg
    time.sleep(.01)
