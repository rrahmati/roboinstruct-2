import theano
import numpy as np
import sys
from theano import tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams
from blocks.model import Model
from blocks.graph import ComputationGraph, apply_batch_normalization, get_batch_normalization_updates
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp, Adam, AdaGrad, AdaDelta, VariableClipping
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing, saveload, ProgressBar
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
# from blocks.extras.extensions.plot import Plot
from utils import get_stream, track_best, MainLoop, Dropout, apply_dropout, SetTrainFlag, load_encoder
from model import nn_fprop
from config import config
from blocks.bricks.conv import ConvolutionalSequence, Convolutional
from blocks.bricks import MLP
from blocks.bricks.recurrent.base import BaseRecurrent
from blocks.roles import PARAMETER, FILTER, INPUT
from blocks import roles
import operator

# Load config parameters
locals().update(config)
# DATA
train_stream = get_stream(hdf5_file, 'train', batch_size)
test_stream = get_stream(hdf5_file, 'test', batch_size)

# MODEL
x = T.TensorType('floatX', [False] * 3)('features')
y = T.tensor3('targets', dtype='floatX')
train_flag = [theano.shared(0)]
x = x.swapaxes(0,1)
y = y.swapaxes(0,1)
out_size = len(output_columns) - 1 if cost_mode == 'RL-MDN' else len(output_columns)
_, latent_size = load_encoder()
in_size = latent_size + len(input_columns)
# mean = x[:,:,0:latent_size]
# var = T.clip(T.exp(x[:,:,latent_size:latent_size*2]), .0001, 1000)
# rrng = MRG_RandomStreams(seed)
# rand = rrng.normal(var.shape, 0, 1, dtype=theano.config.floatX)
# x  = ifelse(T.lt(train_flag[0], .5), T.concatenate([mean , x[:,:,latent_size*2:]], axis=2) , T.concatenate([mean + (var * rand), x[:,:,latent_size*2:]], axis=2))
y_hat, cost, cells = nn_fprop(x, y, in_size, out_size, hidden_size, num_recurrent_layers, train_flag)

# COST
cg = ComputationGraph(cost)
extra_updates = []

# Learning optimizer
if training_optimizer == 'Adam':
    step_rules = [Adam(learning_rate=learning_rate), StepClipping(step_clipping)] # , VariableClipping(threshold=max_norm_threshold)
elif training_optimizer == 'RMSProp':
    step_rules = [RMSProp(learning_rate=learning_rate, decay_rate=decay_rate), StepClipping(step_clipping)]
elif training_optimizer == 'Adagrad':
    step_rules = [AdaGrad(learning_rate=learning_rate), StepClipping(step_clipping)]
elif training_optimizer == 'Adadelta':
    step_rules = [AdaDelta(decay_rate=decay_rate), StepClipping(step_clipping)]

parameters_to_update = cg.parameters
algorithm = GradientDescent(cost=cg.outputs[0], parameters=parameters_to_update,
                            step_rule=CompositeRule(step_rules))
algorithm.add_updates(extra_updates)

# Extensions
gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
step_norm = aggregation.mean(algorithm.total_step_norm)
monitored_vars = [cost, step_rules[0].learning_rate, gradient_norm, step_norm]

test_monitor = DataStreamMonitoring(variables=[cost], after_epoch=True,
                                    before_first_epoch=True, data_stream=test_stream, prefix="test")
train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_epoch=True,
                                       before_first_epoch=True, prefix='train')

set_train_flag = SetTrainFlag(after_epoch=True, before_epoch=True, flag=train_flag)

# plot = Plot('Plotting example', channels=[['cost']], after_batch=True, open_browser=True)
extensions = [set_train_flag, test_monitor, train_monitor, Timing(), Printing(after_epoch=True),
              FinishAfter(after_n_epochs=nepochs),
              saveload.Load(load_path),
              saveload.Checkpoint(last_path,every_n_epochs=10000),
              ] + track_best('test_cost', save_path) #+ track_best('train_cost', last_path)


if learning_rate_decay not in (0, 1):
    extensions.append(SharedVariableModifier(step_rules[0].learning_rate,
                                             lambda n, lr: np.cast[theano.config.floatX](learning_rate_decay * lr),
                                             after_epoch=False, every_n_epochs=lr_decay_every_n_epochs, after_batch=False))

print 'number of parameters in the model: ' + str(T.sum([p.size for p in cg.parameters]).eval())
# Finally build the main loop and train the model
main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     model=Model(cost), extensions=extensions)
main_loop.run()
