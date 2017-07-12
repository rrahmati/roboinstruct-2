import sys, os
import h5py
import numpy as np
from fuel.datasets import H5PYDataset
from fuel.streams import DataStream, ServerDataStream
from fuel.schemes import ShuffledScheme
from fuel.server import start_server
from blocks.extensions import saveload, predicates, SimpleExtension
from blocks.extensions.training import TrackTheBest
from blocks.bricks.interfaces import Activation
from blocks.bricks.base import application
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T
from theano.ifelse import ifelse
from blocks import main_loop
from multiprocessing.process import Process
from fuel.utils import do_not_pickle_attributes
import cv2
import theano
from PIL import Image, ImageChops
from config import config

locals().update(config)
from autoencoder.chainer_vaegan.src.train import encode, decode, get_latent_size

# Define this class to skip serialization of extensions
@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):
    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)

    def load(self):
        self.extensions = []

def apply_dropout(computation_graph, variables, drop_prob):
    rrng = MRG_RandomStreams(seed)
    divisor = (1 - drop_prob)
    replacements = [(var, var *
                     rrng.binomial(var.shape, p=1 - drop_prob,
                                  dtype='floatX'))
                    for var in variables]
    return computation_graph.replace(replacements)

class Dropout(Activation):
    def __init__(self, train_flag, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.flag = train_flag

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        rrng = MRG_RandomStreams(seed=seed)
        print 'dropout'
        return ifelse(T.lt(self.flag[0], .5), input_ , input_ * rrng.binomial(size=input_.shape, p=(1-dropout), dtype='floatX') / (1-dropout))


class SetTrainFlag(SimpleExtension):
    def __init__(self, flag, **kwargs):
        super(SetTrainFlag, self).__init__(**kwargs)
        self.flag = flag

    def do(self, which_callback, *args):
        self.flag[0].set_value(1 - self.flag[0].eval())
        print 'train' if self.flag[0].eval() == 1 else 'evaluate'

class TrajectoryDataset(H5PYDataset):
    filename = hdf5_file

    def __init__(self, which_sets, **kwargs):
        load_in_memory = os.path.getsize(self.filename) < 14 * 10 ** 9 or which_sets == 'test'
        kwargs.setdefault('load_in_memory', load_in_memory)
        super(TrajectoryDataset, self).__init__(file_or_path=self.filename, which_sets=which_sets, **kwargs)

def transpose_stream(data):
    # Required because Recurrent bricks receive as input [sequence, batch, features]
    data = (np.swapaxes(dataset, 0, 1) for dataset in data)
    return data
    # return data

def track_best(channel, save_path):
    sys.setrecursionlimit(1500000)
    tracker = TrackTheBest(channel, choose_best=min)
    checkpoint = saveload.Checkpoint(
        save_path, after_training=False, use_cpickle=True)
    checkpoint.add_condition(["after_epoch"],
                             predicate=predicates.OnLogRecord('{0}_best_so_far'.format(channel)))
    return [tracker, checkpoint]


def get_metadata(hdf5_file):
    with h5py.File(hdf5_file) as f:
        ix_to_out = eval(f['targets'].attrs['ix_to_out'])
        out_to_ix = eval(f['targets'].attrs['out_to_ix'])
        disc_out_size = len(ix_to_out)
    return ix_to_out, out_to_ix, disc_out_size

def load_encoder():
    return None, get_latent_size()

def encode_image(imgs, encoder=None):
    imgs = imgs.astype( theano.config.floatX )
    imgs = imgs /127.5 - 1.
    z, mean, var = encode(imgs)
    return np.concatenate((mean, var), axis=1), mean

def decode_image(z, encoder=None):
    x = decode(z)
    return x

def read_image(path):
    image = Image.open(path)
    return image

def pre_process_image(image, bg_images=None):
    if num_channels == 1:
        image = image.convert('L')
    image = image.resize((image_shape[1], image_shape[0]), Image.ANTIALIAS)
    if num_channels == 3:
        image = image.convert('RGB')
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = image[:, :, ::-1].copy()
    return image

def get_stream(hdf5_file, which_set, batch_size=None):
    dataset = TrajectoryDataset(which_sets=(which_set,))
    if batch_size == None:
        batch_size = dataset.num_examples
    data_stream = DataStream(dataset=dataset, iteration_scheme=ShuffledScheme(
        examples=dataset.num_examples, batch_size=batch_size))

    load_in_memory = os.path.getsize(hdf5_file) < 14 * 10 ** 9 or which_set == 'test'
    if not load_in_memory:
        port = 5557 if which_set == 'train' else 5558
        print port
        server_process = Process(target=start_server, args=(data_stream, port, 10))
        server_process.start()
        data_stream = ServerDataStream(dataset.sources, False, host='localhost', port=port, hwm=10)

    return data_stream
