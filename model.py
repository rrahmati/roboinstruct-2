import theano
import numpy as np
from theano import *
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from blocks import initialization
from blocks.bricks import Linear, Rectifier, cost
from blocks.bricks.parallel import Fork
from architectures import GatedRecurrent, LSTM, LN_LSTM, SimpleRecurrent
from blocks.bricks.cost import AbsoluteError, SquaredError
from blocks.bricks.conv import ConvolutionalSequence, Convolutional, MaxPooling, AveragePooling
from config import config
from blocks.initialization import Constant, Uniform
from blocks.bricks import MLP, BatchNormalizedMLP, BatchNormalization, Logistic, Initializable, FeedforwardSequence, Tanh
from utils import get_metadata, Dropout
import logging

locals().update(config)

def initialize(to_init, weights_init=Uniform(width=0.08), biases_init=Constant(0)):
    for bricks in to_init:
        bricks.weights_init = weights_init
        bricks.biases_init = biases_init
        bricks.initialize()


def MDN_output_layer(x, h, y, in_size, out_size, hidden_size, pred, task, components_size=components_size):
    if connect_h_to_o:
        hiddens = T.concatenate([hidden for hidden in h], axis=2)
        hidden_out_size = hidden_size * len(h)
    else:
        hiddens = h[-1]
        hidden_out_size = hidden_size

    n_comps = components_size
    mu_linear = Linear(name='mu_linear' + str(pred), input_dim=hidden_out_size, output_dim=out_size * n_comps)
    sigma_linear = Linear(name='sigma_linear' + str(pred), input_dim=hidden_out_size, output_dim=n_comps)
    mixing_linear = Linear(name='mixing_linear' + str(pred), input_dim=hidden_out_size, output_dim=n_comps)
    initialize([mu_linear, sigma_linear, mixing_linear])
    mu = mu_linear.apply(hiddens)
    mu = mu.reshape((mu.shape[0], mu.shape[1], out_size, n_comps))
    sigma_orig = sigma_linear.apply(hiddens)
    mixing_orig = mixing_linear.apply(hiddens)
    sigma = T.nnet.softplus(sigma_orig)
    # apply softmax to mixing
    e_x = T.exp(mixing_orig - mixing_orig.max(axis=2, keepdims=True))
    mixing = e_x / e_x.sum(axis=2, keepdims=True)

    # calculate cost
    exponent = -0.5 * T.inv(sigma) * T.sum((y.dimshuffle(0, 1, 2, 'x') - mu) ** 2, axis=2)
    normalizer = (2 * np.pi * sigma)
    exponent = exponent + T.log(mixing) - (out_size * .5) * T.log(normalizer)
    # LogSumExp(x)
    max_exponent = T.max(exponent, axis=2, keepdims=True)
    mod_exponent = exponent - max_exponent
    gauss_mix = T.sum(T.exp(mod_exponent), axis=2, keepdims=True)
    log_gauss = T.log(gauss_mix) + max_exponent
    cost = -log_gauss

    # sampling
    srng = RandomStreams(seed=seed)
    mixing = mixing_orig * (1 + sampling_bias)
    sigma = T.nnet.softplus(sigma_orig - sampling_bias)
    e_x = T.exp(mixing - mixing.max(axis=2, keepdims=True))
    mixing = e_x / e_x.sum(axis=2, keepdims=True)
    component = srng.multinomial(pvals=mixing)
    component_mean = T.sum(mu * component.dimshuffle(0, 1, 'x', 2), axis=3)
    component_std = T.sum(sigma * component, axis=2, keepdims=True)
    linear_output = srng.normal(avg=component_mean, std=component_std)
    linear_output.name = 'linear_output'

    return linear_output, cost

def RL_MDN_output_layer(x, h, y, in_size, out_size, hidden_size, pred, task, components_size=components_size):
    advantage = T.pow(gamma, y[:,:,-1:]) #- v_s
    y = y[:,:,:out_size]
    if single_dim_out:
        advantage = T.extra_ops.repeat(advantage, out_size, axis=0) * out_size
        y = y.swapaxes(0, 1)
        y = y.reshape((y.shape[0], y.shape[1]*out_size, 1))
        y = y.swapaxes(0, 1)
        out_size = 1
    linear_output, cost = MDN_output_layer(x, h, y, in_size, out_size, hidden_size, pred, task, components_size)
    cost = T.mul(cost, advantage)
    return linear_output, cost

def MSE_output_layer(x, h, y, in_size, out_size, hidden_size, pred, task):
    if connect_h_to_o:
        hiddens = T.concatenate([hidden for hidden in h], axis=2)
        hidden_out_size = hidden_size *len(h)
        hidden_to_output = Linear(name='hidden_to_output' + str(pred), input_dim=hidden_out_size,
                                  output_dim=out_size)
    else:
        hidden_to_output = Linear(name='hidden_to_output' + str(pred), input_dim=hidden_size,
                                  output_dim=out_size)
        hiddens = h[-1]
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(hiddens)
    linear_output.name = 'linear_output'
    cost = T.sqr(y - linear_output).mean() # + T.mul(T.sum(y[:,:,8:9],axis=1).mean(),2)
    return linear_output, cost

def output_layer(x, hiddens, y, in_size, out_size, hidden_size, components_size=components_size):
    costs = []
    linear_outputs = []
    for i in range(len(future_predictions)):
        linear_outputs.append([])
        if cost_mode == 'RL-MDN':
            linear_output, cost = RL_MDN_output_layer(x, hiddens, y[:, :, (out_size+1) * i:(out_size+1) * (i + 1)], in_size,
                                                   out_size, hidden_size, str(i), components_size)
        if cost_mode == 'MDN':
            linear_output, cost = MDN_output_layer(x, hiddens, y[:, :, out_size * i:out_size * (i + 1)], in_size,
                                                   out_size, hidden_size, str(i))
        elif cost_mode == 'MSE':
            linear_output, cost = MSE_output_layer(x, hiddens, y[:, :, out_size * i:out_size * (i + 1)], in_size,
                                                   out_size, hidden_size, str(i))
        linear_outputs[i].append(linear_output)
        costs.append(T.mul(cost, prediction_cost_weights[i]))
    cost_sum = T.mean(costs)
    cost_sum.name = 'cost'
    return linear_outputs[0][0], cost_sum


def linear_layer(in_size, dim, x, h, n, first_layer=False):
    if first_layer:
        input = x
        linear = Linear(input_dim=in_size, output_dim=dim, name='feedforward' + str(n))
    elif connect_x_to_h:
        input = T.concatenate([x] + [h[n - 1]], axis=1)
        linear = Linear(input_dim=in_size + dim, output_dim=dim, name='feedforward' + str(n))
    else:
        input = h[n - 1]
        linear = Linear(input_dim=dim, output_dim=dim, name='feedforward' + str(n))
    initialize([linear])
    return linear.apply(input)

def rnn_layer(in_size, dim, x, h, n, first_layer = False):
    if connect_h_to_h == 'all-previous':
        if first_layer:
            rnn_input = x
            linear = Linear(input_dim=in_size, output_dim=dim, name='linear' + str(n))
        elif connect_x_to_h:
            rnn_input = T.concatenate([x] + [hidden for hidden in h], axis=2)
            linear = Linear(input_dim=in_size + dim * n, output_dim=dim, name='linear' + str(n))
        else:
            rnn_input = T.concatenate([hidden for hidden in h], axis=2)
            linear = Linear(input_dim=dim * n, output_dim=dim, name='linear' + str(n))
    elif connect_h_to_h == 'two-previous':
        if first_layer:
            rnn_input = x
            linear = Linear(input_dim=in_size, output_dim=dim, name='linear' + str(n))
        elif connect_x_to_h:
            rnn_input = T.concatenate([x] + h[max(0, n - 2):n], axis=2)
            linear = Linear(input_dim=in_size + dim * 2 if n > 1 else in_size + dim, output_dim=dim, name='linear' + str(n))
        else:
            rnn_input = T.concatenate(h[max(0, n - 2):n], axis=2)
            linear = Linear(input_dim=dim * 2 if n > 1 else dim, output_dim=dim, name='linear' + str(n))
    elif connect_h_to_h == 'one-previous':
        if first_layer:
            rnn_input = x
            linear = Linear(input_dim=in_size, output_dim=dim, name='linear' + str(n))
        elif connect_x_to_h:
            rnn_input = T.concatenate([x] + [h[n-1]], axis=2)
            linear = Linear(input_dim=in_size + dim, output_dim=dim, name='linear' + str(n))
        else:
            rnn_input = h[n]
            linear = Linear(input_dim=dim, output_dim=dim, name='linear' + str(n))
    rnn = SimpleRecurrent(dim=dim, activation=Tanh(), name=layer_models[n] + str(n))
    initialize([linear, rnn])
    if layer_models[n] == 'rnn':
        return rnn.apply(linear.apply(lstm_input))
    elif layer_models[n] == 'mt_rnn':
        return rnn.apply(linear.apply(rnn_input), time_scale=layer_resolutions[n], time_offset=layer_execution_time_offset[n])

def lstm_layer(in_size, dim, x, h, n, first_layer = False):
    if connect_h_to_h == 'all-previous':
        if first_layer:
            lstm_input = x
            linear = Linear(input_dim=in_size, output_dim=dim * 4, name='linear' + str(n))
        elif connect_x_to_h:
            lstm_input = T.concatenate([x] + [hidden for hidden in h], axis=2)
            linear = Linear(input_dim=in_size + dim * (n), output_dim=dim * 4, name='linear' + str(n))
        else:
            lstm_input = T.concatenate([hidden for hidden in h], axis=2)
            linear = Linear(input_dim=dim * (n + 1), output_dim=dim * 4, name='linear' + str(n))
    elif connect_h_to_h == 'two-previous':
        if first_layer:
            lstm_input = x
            linear = Linear(input_dim=in_size, output_dim=dim * 4, name='linear' + str(n))
        elif connect_x_to_h:
            lstm_input = T.concatenate([x] + h[max(0, n - 2):n], axis=2)
            linear = Linear(input_dim=in_size + dim * 2 if n > 1 else in_size + dim, output_dim=dim * 4, name='linear' + str(n))
        else:
            lstm_input = T.concatenate(h[max(0, n - 2):n], axis=2)
            linear = Linear(input_dim=dim * 2 if n > 1 else dim, output_dim=dim * 4, name='linear' + str(n))
    elif connect_h_to_h == 'one-previous':
        if first_layer:
            lstm_input = x
            linear = Linear(input_dim=in_size, output_dim=dim * 4, name='linear' + str(n))
        elif connect_x_to_h:
            lstm_input = T.concatenate([x] + [h[n-1]], axis=2)
            linear = Linear(input_dim=in_size + dim, output_dim=dim * 4, name='linear' + str(n))
        else:
            lstm_input = h[n-1]
            linear = Linear(input_dim=dim, output_dim=dim * 4, name='linear' + str(n))
    if use_layer_norm:
        lstm = LN_LSTM(dim=dim , name=layer_models[n] + str(n))
    else:
        lstm = LSTM(dim=dim , name=layer_models[n] + str(n))
    initialize([linear, lstm])
    if layer_models[n] == 'lstm':
        return lstm.apply(linear.apply(lstm_input))
    elif layer_models[n] == 'mt_lstm':
        return lstm.apply(linear.apply(lstm_input), time_scale=layer_resolutions[n], time_offset=layer_execution_time_offset[n])


def add_layer(model, i, in_size, h_size, x, h, cells, train_flag, first_layer = False):
    if dropout > 0 and not first_layer:
        h[i-1] = Dropout(name='dropout__recurrent_hidden{}'.format(i), train_flag=train_flag).apply(h[i-1])
    print h_size
    if model == 'rnn' or model == 'mt_rnn':
        h.append(rnn_layer(in_size, h_size, x, h, i, first_layer))
    if model == 'lstm' or model == 'mt_lstm':
        state, cell = lstm_layer(in_size, h_size, x, h, i, first_layer)
        h.append(state)
        cells.append(cell)
    if model == 'feedforward':
        h.append(linear_layer(in_size, h_size, x, h, i, first_layer))
    return h, cells

def task_ID_layers(x, recurrent_in_size):
    mlp = MLP([Rectifier()] * (len(task_ID_FF_dims)-1), task_ID_FF_dims, name='task_ID_mlp', weights_init=Uniform(width=.2), biases_init=Constant(0))
    mlp.push_initialization_config()
    mlp.initialize()
    out_size = task_ID_FF_dims[-1] + recurrent_in_size - len(game_tasks)
    zero_padded_task_IDs = T.concatenate([x[:,:,-len(game_tasks):], T.zeros((x.shape[0], x.shape[1], task_ID_FF_dims[0] - len(game_tasks)))], axis=2)
    mlp_out = mlp.apply(zero_padded_task_IDs)
    task_ID_out = T.concatenate([x[:,:,:-len(game_tasks)]] + [mlp_out], axis=2)
    return task_ID_out, out_size

def nn_fprop(x, y, recurrent_in_size, out_size, hidden_size, num_recurrent_layers, train_flag):
    if task_ID_type == 'feedforward':
        x, recurrent_in_size = task_ID_layers(x, recurrent_in_size)
    recurrent_input = x
    cells = []
    h = []
    if dropout > 0:
        recurrent_input = Dropout(name='dropout_recurrent_in', train_flag=train_flag).apply(recurrent_input)
    if linear_before_recurrent_size > 0:
        linear = Linear(input_dim=2, output_dim=linear_before_recurrent_size, name='linear_befor_recurrent')
        initialize([linear])
        recurrent_input = linear.apply(recurrent_input[:,:,-2:])
        recurrent_in_size = linear_before_recurrent_size
    if single_dim_out:
        recurrent_input = T.extra_ops.repeat(recurrent_input, out_size, axis=0)
    p_components_size = components_size
    for i in range(num_recurrent_layers):
        model = layer_models[i]
        h, cells = add_layer(model, i, recurrent_in_size, hidden_size, recurrent_input, h, cells, train_flag,
                             first_layer=True if i == 0 else False)
    return output_layer(recurrent_input, h, y, recurrent_in_size, out_size, hidden_size, p_components_size) + (cells,)
