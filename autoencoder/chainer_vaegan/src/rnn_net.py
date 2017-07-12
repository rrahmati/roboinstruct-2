import numpy as np
import math
from chainer import training, optimizers, serializers, utils, datasets, iterators, report
from chainer import Variable, Link, Chain, ChainList
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L


class MDN_RNN(Chain):
    """
    Mixure density network model.
    """
    def __init__(self, IN_DIM, HIDDEN_DIM, OUT_DIM, NUM_MIXTURE):
        self.IN_DIM = IN_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.OUT_DIM = OUT_DIM
        self.NUM_MIXTURE = NUM_MIXTURE
        super(MDN_RNN, self).__init__(
            l1_ = L.LSTM(IN_DIM, HIDDEN_DIM),
            mixing_ = L.Linear(HIDDEN_DIM, NUM_MIXTURE),
            mu_ = L.Linear(HIDDEN_DIM, NUM_MIXTURE*OUT_DIM),
            sigma_ = L.Linear(HIDDEN_DIM, NUM_MIXTURE)
        )

    def __call__(self, x, y):
        h = self.l1_(x)
        # h2 = self.l2_(h1)
        # return F.mean_squared_error(h2, y)

        sigma = F.softplus(self.sigma_(h))
        mixing = F.softmax(self.mixing_(h))
        mu = F.reshape(self.mu_(h), (-1, self.OUT_DIM, self.NUM_MIXTURE))
        mu, y = F.broadcast(mu, F.reshape(y, (-1, self.OUT_DIM, 1)))
        exponent = -0.5 * (1. / sigma) * F.sum((y - mu) ** 2, axis=1)
        normalizer = 2 * np.pi * sigma
        exponent = exponent + F.log(mixing) - (self.OUT_DIM * .5) * F.log(normalizer)
        cost = -F.logsumexp(exponent)
        return cost

        # coef = F.softmax(self.coef_(h))
        # mean = F.reshape(self.mean_(h), (-1,self.NUM_MIXTURE,self.OUT_DIM))
        # logvar = F.exp(self.logvar_(h))
        # mean, y = F.broadcast(mean, F.reshape(y, (-1,1,self.OUT_DIM)))
        # return F.sum(
        #     coef*F.exp(-0.5*F.sum((y-mean)**2, axis=2)*F.exp(-logvar))/
        #     ((2*np.pi*F.exp(logvar))**(0.5*self.OUT_DIM)),axis=1)

    def reset_state(self):
        self.l1_.reset_state()
