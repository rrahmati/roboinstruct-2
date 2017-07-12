import numpy as np
import theano
import theano.d3viz as d3v
from theano import tensor
from blocks import roles
from blocks.roles import OUTPUT
from blocks.model import Model
from blocks.extensions import saveload
from blocks.filter import VariableFilter
from utils import MainLoop
from config import config
from model import nn_fprop
from utils import pre_process_image, load_encoder, encode_image, decode_image
import argparse
import sys
import os
import pandas as pd
import time
import signal
from pandas.parser import CParserError
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy import dtype
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import PIL
import scipy

locals().update(config)
sceneStateFile = os.path.abspath("predictions/sceneState")

def load_models(model_path=save_path, in_size=len(input_columns),
                out_size=len(output_columns) - 1 if cost_mode == 'RL-MDN' else len(output_columns),
                hidden_size=hidden_size, num_recurrent_layers=num_recurrent_layers, model=layer_models[0]):
    initials = []
    if not os.path.isfile(model_path):
        print 'Could not find model file.'
        sys.exit(0)
    print 'Loading model from {0}...'.format(model_path)
    x = tensor.tensor3('features', dtype=theano.config.floatX)
    y = tensor.tensor3('targets', dtype='floatX')
    train_flag = [theano.shared(0)]
    _, latent_size = load_encoder()
    in_size = latent_size + len(input_columns)
    y_hat, cost, cells = nn_fprop(x, y, in_size, out_size, hidden_size, num_recurrent_layers, train_flag)
    main_loop = MainLoop(algorithm=None, data_stream=None, model=Model(cost),
                         extensions=[saveload.Load(model_path)])
    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')
    bin_model = main_loop.model
    print 'Model loaded. Building prediction function...'
    hiddens = []
    for i in range(num_recurrent_layers):
        brick = [b for b in bin_model.get_top_bricks() if b.name == layer_models[i] + str(i)][0]
        hiddens.extend(VariableFilter(theano_name=brick.name + '_apply_states')(bin_model.variables))
        hiddens.extend(VariableFilter(theano_name=brick.name + '_apply_cells')(cells))
        initials.extend(VariableFilter(roles=[roles.INITIAL_STATE])(brick.parameters))
    predict_func = theano.function([x], hiddens + [y_hat])
    encoder, code_size = load_encoder()
    return predict_func, initials, encoder, code_size


def predict_one_timestep(predict_func, encoder, code_size, initials, x, out_size, iteration):
    try:
        img = CvBridge().imgmsg_to_cv2(camera1_msg, "bgr8")
        img = np.array(img, dtype=np.float)
        img = img[0:540, 250:840]
        cv2.imwrite('predictions/current_image.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    except (CvBridgeError) as e:
        print(e)
    else:
        image = PIL.Image.open('predictions/current_image.jpg')
        current_scene_image = pre_process_image(image)
        cv2.imshow('Input image', cv2.resize(np.array(current_scene_image.transpose((1, 2, 0)))[...,::-1], (0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST ))
        cv2.waitKey(10)
    current_scene_image = np.array(current_scene_image, dtype=np.float32)
    images = np.array([current_scene_image])
    _, encoded_images= encode_image(images, encoder)
    decoded_images = decode_image(encoded_images, encoder)
    cv2.imshow('Reconstructed image', cv2.resize(np.array(decoded_images[0].transpose((1, 2, 0)))[...,::-1], (0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST ))
    cv2.waitKey(10)
    x = np.concatenate([encoded_images[0], x])
    newinitials = predict_func([[x]])
    raw_prediction = newinitials.pop().astype(theano.config.floatX)
    if single_dim_out:
        predicted_values = raw_prediction[:, -1, -1].astype(theano.config.floatX).reshape((len(raw_prediction),))
    else:
        predicted_values = raw_prediction[-1, -1, :].astype(theano.config.floatX)
    layer = 0
    for initial, newinitial in zip(initials, newinitials):
        if iteration % layer_resolutions[layer // 2] == 0:
            initial.set_value(newinitial[-1].flatten())
        layer += (2 if layer_models[layer // 2] == 'mt_rnn' else 1)
        layer = min([layer, len(layer_resolutions)])
    return predicted_values, newinitials


def set_task_column_to_one_hot(data):
    if config['multi_task_mode'] == 'ID':
        for i in config['game_tasks']:
            data['task' + str(i)] = 0
            data.loc[data['task'] == i, 'task' + str(i)] = 1
    return data

def plot_arrays(arrays, title='image'):
    images = []
    for i in range(int(len(arrays)/8)):
        images.append(np.hstack(arrays[i*8:(i+1)*8]))
    images = np.vstack(images)
    vis = cv2.cvtColor(np.array(images, np.float32), cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (0,0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST )
    cv2.imshow(title, vis)
    cv2.waitKey(10)

def sample():
    if plot_hidden_states:
        plt.ion()
        plt.ylim([-2, +4])
        plt.show()
    predict_func, initials, encoder, code_size = load_models()
    print("Generating trajectory...")
    last_time = 0
    counter = 0
    out_size = len(output_columns) - 1 if cost_mode == 'RL-MDN' else len(output_columns)
    last_speed_calc = time.time()
    predicted = np.array([0.749, 0.785, 0.613, 0.459, 0.679, 1., 0.])
    last_prediction = predicted.copy()
    hidden_states = np.empty((num_recurrent_layers, hidden_size), dtype='float32')
    active_hidden_states = np.empty((num_recurrent_layers, hidden_size), dtype='float32')
    for iteration in range(10000000):
        try:
            try:
                command_msg = Float32MultiArray()
                command_msg.data = predicted[0:out_size]
                print predicted
                robot_command_pub.publish(command_msg)
            except IOError:
                print 'could not open the prediction file.'
            prediction_diff = ((last_prediction[0:out_size] - predicted[0:out_size]) ** 2).mean()
            min_wait = np.clip(0.2 + prediction_diff * 80, .2, .5)
            time.sleep(min_wait)
            while True:
                new_state = pd.DataFrame({'task': [task_to_perform], 'time': [time.time()], 'gripper': [predicted[0]], 'joint1': [predicted[1]],
                    'joint2': [predicted[2]], 'joint3': [predicted[3]], 'joint4': [predicted[4]], 'joint5': [predicted[5]]})
                new_state = set_task_column_to_one_hot(new_state)
                print np.array(new_state[input_columns].iloc[0], dtype=theano.config.floatX)
                if last_time == new_state['time'][0]:
                    time.sleep(.005)
                    continue
                else:
                    break
            last_time = new_state['time'][0]
            x = np.array(new_state[input_columns].iloc[0], dtype=theano.config.floatX)
            predicted, newinitials = predict_one_timestep(predict_func, encoder, code_size, initials, x, out_size, iteration)
            last_prediction = predicted.copy()
            if plot_hidden_states:
                plot_arrays(newinitials)
        except(RuntimeError):
            print sys.exc_info()[0]

        counter += 1
        if (time.time() - last_speed_calc > 1):
            counter = 0
            last_speed_calc = time.time()


if __name__ == '__main__':
    if robot == 'al5d' or robot == 'mico':
        rospy.init_node('roboinstruct')
        def camera1_callback(msg):
            global camera1_msg
            camera1_msg = msg

        def move_callback(msg):
            global move_msg
            global last_move_msg_time
            move_msg = msg
            last_move_msg_time = time.time()

        image_sub = rospy.Subscriber(camera1_image_topic, Image, camera1_callback)
        image_sub = rospy.Subscriber("/move_info_for_test", Float32MultiArray, move_callback)
        robot_command_pub = rospy.Publisher("/robot_command", Float32MultiArray, queue_size=100)

        def signal_handler(signal, frame):
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

    locals().update(config)
    float_formatter = lambda x: "%.5f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})
    sample()
