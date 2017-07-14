import theano
import numpy as np
import sys
import cv2
import h5py
from fuel.datasets import H5PYDataset
from config import config
import os
import pandas as pd
from sample import set_task_column_to_one_hot

# Load config parameters
locals().update(config)
from utils import read_image, pre_process_image, load_encoder, encode_image
from PIL import Image
np.random.seed(0)

def setDataResolution(dataCopy):
    data = pd.DataFrame()
    for index_offset in xrange(0, waypoint_resolution, resolution_step):
        indexes = xrange(index_offset, len(dataCopy.index), waypoint_resolution)
        data = data.append(pd.DataFrame(dataCopy.iloc[indexes]))
        # print len(data)
    data = data.reset_index()
    return data


def setStepsToGoal(data):
    dataCopy = data.copy()
    if 'reward' in data.columns:
        data['steps_to_goal'] = data['reward']
    goal_reached_indices = data.loc[data['steps_to_goal'] > .5, 'steps_to_goal'].index.tolist()
    goal_reached_indices = [min(data.index.tolist())] + goal_reached_indices
    for index in range(len(goal_reached_indices)-1):
        index_1 = goal_reached_indices[index]
        index_2 = goal_reached_indices[index+1]
        data.ix[xrange(index_1,index_2),'steps_to_goal'] = xrange(index_2-index_1, 0, -1)
    print 'How many times reward given: ', len(goal_reached_indices)
    # data = data.loc[data['steps_to_goal'] != 500]
    return data

def pre_process_data(data):
    data['steps_to_goal'] = 500
    if not 'reward' in data.columns:
        data['reward'] = 0
    data = set_task_column_to_one_hot(data)
    data = pd.DataFrame(data[list(set(
        ['task', 'reward'] + input_columns + list(
            set(output_columns) - set(
                input_columns))))])
    if cost_mode == 'RL-MDN':
        setStepsToGoal(data)
    data = setDataResolution(data.copy())
    data = pd.DataFrame(data[input_columns + list(
        set(output_columns) - set(
            input_columns))])
    return data


def train_test_split():
    print("Loading data...")
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    train_img_data = np.memmap('train_imgs.npy', dtype='uint8', mode='w+',
                    shape=(1, num_channels, image_shape[0], image_shape[1]))
    test_img_data = np.memmap('test_imgs.npy', dtype='uint8', mode='w+',
                    shape=(1, num_channels, image_shape[0], image_shape[1]))
    img_data = np.memmap('imgs.npy', dtype='uint8', mode='w+',
                              shape=(1, num_channels, image_shape[0], image_shape[1]))
    counter = 0
    max_files = 2000
    task_data_loaded = [0] * len(game_tasks)
    for i in xrange(len(game_tasks)):
        path = trajs_paths + '/task-' + str(game_tasks[i])
        for filename in [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]:
            print path, filename
            counter += 1
            if counter > max_files:
                break
            new_data = pd.read_csv(os.path.join(path, filename), header=2, sep=',', index_col=False)
            task_data_loaded[i] += len(new_data.index)
            # new_data = new_data[:200]
            if len(new_data.index) == 0:
                continue
            new_data_size = len(new_data.index)
            new_train = pd.DataFrame(new_data.iloc[0:int(new_data_size * train_size)])
            new_test = pd.DataFrame(new_data.iloc[int(new_data_size * train_size):new_data_size])
            new_train = pre_process_data(new_train)
            new_test = pre_process_data(new_test)
            if make_dataset_mode == 'prepare_data_for_chainer':
                new_train_img_data = read_images(trajs_paths + '/task-' + str(game_tasks[i]) + '/' + os.path.splitext(filename)[0] + '/camera-1', 0, int(new_data_size * train_size))
                new_test_img_data = read_images(trajs_paths + '/task-' + str(game_tasks[i]) + '/' + os.path.splitext(filename)[0] + '/camera-1', int(new_data_size * train_size), new_data_size)
            for j in xrange(task_weights[i]):
                train_data = train_data.append(new_train)
                test_data = test_data.append(new_test)
                if make_dataset_mode == 'prepare_data_for_chainer':
                    train_img_data = np.memmap('train_imgs.npy', dtype='uint8', mode='r+',
                                               shape=(len(train_img_data)+len(new_train_img_data), num_channels, image_shape[0], image_shape[1]), order='C')
                    train_img_data[-len(new_train_img_data):,:,:,:] = new_train_img_data
                    test_img_data = np.memmap('test_imgs.npy', dtype='uint8', mode='r+',
                                               shape=(len(test_img_data)+len(new_test_img_data), num_channels, image_shape[0], image_shape[1]), order='C')
                    test_img_data[-len(new_test_img_data):,:,:,:] = new_test_img_data
                    os.remove(new_train_img_data.filename)
                    os.remove(new_test_img_data.filename)
    train_data_in = pd.DataFrame(train_data[input_columns])
    train_data_out = pd.DataFrame(train_data[output_columns])
    test_data_in = pd.DataFrame(test_data[input_columns])
    test_data_out = pd.DataFrame(test_data[output_columns])

    train_data_size = len(train_data_in.index)
    data_in = train_data_in.append(test_data_in)
    data_out = train_data_out.append(test_data_out)
    if make_dataset_mode == 'prepare_data_for_chainer':
        img_data = np.memmap('imgs.npy', dtype='uint8', mode='w+',
                                   shape=(len(train_img_data)+len(test_img_data)-2, num_channels, image_shape[0], image_shape[1]))
        img_data[0:len(train_img_data)-1, :, :, :] = train_img_data[1:]
        img_data[len(train_img_data)-1:, :, :, :] = test_img_data[1:]
        os.remove(train_img_data.filename)
        os.remove(test_img_data.filename)

    print "input columns: ", input_columns
    print "output_columns: ", output_columns

    print data_in.shape, data_out.shape, img_data.shape
    return data_in.as_matrix(), data_out.as_matrix(), img_data, train_data_size

def getConvFeatures(data_in, img_data):
    n_batch = 100
    encoder, code_size = load_encoder()
    img_features = np.empty((len(data_in), code_size), dtype=theano.config.floatX)
    for i in xrange(len(data_in)/n_batch+1):
        sys.stdout.write('\r' + str(i) + '/' + str(len(data_in)/n_batch))
        sys.stdout.flush()  # important
        start = i*n_batch
        end = min((i+1)*n_batch,len(data_in))
        images = img_data[start:end]
        _, img_features[start:end] = encode_image(images, encoder)
    data_in = np.column_stack((img_features, data_in))
    return data_in, data_in.shape[1]


def read_images(path, start_index, end_index):
    """Extract the images into a 4D tensor [image index, channels, y, x].
    """
    data = np.memmap(str(np.random.randint(1000000)) + '.npy', dtype='uint8', mode='w+',
                     shape=(end_index-start_index, num_channels, image_shape[0], image_shape[1]))
    sample_index = 0
    for index_offset in xrange(0, waypoint_resolution, resolution_step):
        for j in xrange(start_index + index_offset, end_index, waypoint_resolution):
            if sample_index % 1000 == 0:
                sys.stdout.write('\r' + str(sample_index) + ' / ' + str((end_index-start_index)))
                sys.stdout.flush()  # important
            image = read_image(path + '/' + str(j) + ".jpg")
            data[sample_index] = pre_process_image(image)
            sample_index += 1
    print ''
    return data


def main():
    data_in, data_out, img_data, train_data_size = train_test_split()
    if make_dataset_mode == 'prepare_data_for_chainer':
        np.save('img_data.npy', img_data)
        np.save('data_in.npy', data_in)
        np.save('data_out.npy', data_out)
        print 'Files saved on disc for Chainer.'
    elif make_dataset_mode == 'prepare_data_for_blocks':
        img_data = np.load('img_data.npy', mmap_mode='r')
        data_in, in_size = getConvFeatures(data_in, img_data)
        out_size = len(output_columns)
        max_prediction = max(future_predictions) + 1
        if len(data_in) % seq_length > 0:
            data_in = data_in[:len(data_in) - len(data_in) % seq_length + max_prediction]
        else:
            data_in = data_in[:len(data_in) - seq_length + max_prediction]
        nsamples = (len(data_in) / seq_redundancy)
        print 'Saving data to disc...'
        inputs = np.memmap('inputs.npy', dtype=theano.config.floatX, mode='w+', shape=(nsamples, seq_length, in_size))
        outputs = np.memmap('outputs.npy', dtype=theano.config.floatX, mode='w+',
                            shape=(nsamples, seq_length, len(future_predictions) * out_size))
        for i, p in enumerate(xrange(0, len(data_in) - max_prediction - seq_length, seq_redundancy)):
            inputs[i] = np.array([d for d in data_in[p:p + seq_length]])
            for j in xrange(len(future_predictions)):
                outputs[i, :, j * out_size:(j + 1) * out_size] = np.array(
                    [d for d in data_out[p + future_predictions[j]:p + seq_length + future_predictions[j]]])

        nsamples = len(inputs)
        nsamples_train = train_data_size // seq_length

        print np.isnan(np.sum(inputs))
        print np.isnan(np.sum(outputs))

        f = h5py.File(hdf5_file, mode='w')
        features = f.create_dataset('features', inputs.shape, dtype=theano.config.floatX)
        targets = f.create_dataset('targets', outputs.shape, dtype=theano.config.floatX)

        features[...] = inputs
        targets[...] = outputs
        features.dims[0].label = 'batch'
        features.dims[1].label = 'sequence'
        features.dims[2].label = 'features'
        targets.dims[0].label = 'batch'
        targets.dims[1].label = 'sequence'
        targets.dims[2].label = 'outputs'
        split_dict = {
            'train': {'features': (0, nsamples_train), 'targets': (0, nsamples_train)},
            'test': {'features': (nsamples_train, nsamples), 'targets': (nsamples_train, nsamples)}}
        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        f.flush()
        f.close()
        print 'inputs shape:', inputs.shape
        print 'outputs shape:', outputs.shape
        print 'image inputs shape:', img_data.shape
        os.remove(img_data.filename)
        os.remove(inputs.filename)
        os.remove(outputs.filename)
        print 'Files saved on disc for Blocks!'

if __name__ == "__main__":
    main()
