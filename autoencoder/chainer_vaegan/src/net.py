import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal


class Encoder(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Encoder, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            mean=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                          initialW=Normal(0.02)),
            var=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))
            mean = self.mean(h4)
            var = self.var(h4)
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.clip(F.exp(var), .001, 100.) * Variable(rand)
            # z  = mean + F.exp(var) * Variable(rand, volatile=not train)
            return z, mean, var, h4


class Generator(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Generator, self).__init__(
            g1=L.Linear(latent_size, initial_size * initial_size * int(128 * density),
                        initialW=Normal(0.02)),
            norm1=L.BatchNormalization(initial_size * initial_size * int(128 * density)),
            g2=L.Deconvolution2D(int(128 * density), int(64 * density), 4, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(64 * density)),
            g3=L.Deconvolution2D(int(64 * density), int(32 * density), 4, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(32 * density)),
            g4=L.Deconvolution2D(int(32 * density), int(16 * density), 4, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(16 * density)),
            g5=L.Deconvolution2D(int(16 * density), channel, 4, stride=2, pad=1,
                                 initialW=Normal(0.02)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        with chainer.using_config('train', train):
            h1 = F.reshape(F.relu(self.norm1(self.g1(z))),
                           (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h3 = F.relu(self.norm3(self.g3(h2)))
            h4 = F.relu(self.norm4(self.g4(h3)))
            return F.tanh(self.g5(h4))


class EncoderDeep(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(EncoderDeep, self).__init__(
            dc1=L.Convolution2D(channel, 16 * density, 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(16 * density, 32 * density, 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(32 * density),
            dc3=L.Convolution2D(32 * density, 64 * density, 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(64 * density),
            dc4=L.Convolution2D(64 * density, 128 * density, 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(128 * density),

            dc1_=L.Convolution2D(16 * density, 16 * density, 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            dc2_=L.Convolution2D(32 * density, 32 * density, 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm2_=L.BatchNormalization(32 * density),
            dc3_=L.Convolution2D(64 * density, 64 * density, 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(64 * density),
            dc4_=L.Convolution2D(128 * density, 128 * density, 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm4_=L.BatchNormalization(128 * density),

            mean=L.Linear(initial_size * initial_size * 128 * density, latent_size,
                          initialW=Normal(0.02)),
            var=L.Linear(initial_size * initial_size * 128 * density, latent_size,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train):
            xp = cuda.get_array_module(x.data)
            h1 = F.leaky_relu(self.dc1(x))
            h1_ = F.leaky_relu(self.dc1_(h1))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1_)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h4 = F.leaky_relu(self.norm4_(self.dc4(h3_)))
            h4_ = F.leaky_relu(self.norm4_(self.dc4_(h4)))
            mean = self.mean(h4_)
            var = self.var(h4_)
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand, volatile=not train)
            return (z, mean, var)


class GeneratorDeep(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(GeneratorDeep, self).__init__(
            g1=L.Linear(latent_size, initial_size * initial_size * 128 * density, initialW=Normal(0.02)),
            norm1=L.BatchNormalization(initial_size * initial_size * 128 * density),
            g2=L.Deconvolution2D(128 * density, 64 * density, 4, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm2=L.BatchNormalization(64 * density),
            g3=L.Deconvolution2D(64 * density, 32 * density, 4, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm3=L.BatchNormalization(32 * density),
            g4=L.Deconvolution2D(32 * density, 16 * density, 4, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm4=L.BatchNormalization(16 * density),
            g5=L.Deconvolution2D(16 * density, channel, 4, stride=2, pad=1,
                                 initialW=Normal(0.02)),

            g2_=L.Deconvolution2D(64 * density, 64 * density, 3, stride=1, pad=1,
                                  initialW=Normal(0.02)),
            norm2_=L.BatchNormalization(64 * density),
            g3_=L.Deconvolution2D(32 * density, 32 * density, 3, stride=1, pad=1,
                                  initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(32 * density),
            g4_=L.Deconvolution2D(16 * density, 16 * density, 3, stride=1, pad=1,
                                  initialW=Normal(0.02)),
            norm4_=L.BatchNormalization(16 * density),
            g5_=L.Deconvolution2D(channel, channel, 3, stride=1, pad=1, initialW=Normal(0.02)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        with chainer.using_config('train', train):
            h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)),
                           (z.data.shape[0], 128 * self.density, self.initial_size, self.initial_size))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h2_ = F.relu(self.norm2_(self.g2_(h2)))
            h3 = F.relu(self.norm3(self.g3(h2_)))
            h3_ = F.relu(self.norm3_(self.g3_(h3)))
            h4 = F.relu(self.norm4(self.g4(h3_)))
            h4_ = F.relu(self.norm4_(self.g4_(h4)))
            return F.tanh(self.g5(h4_))


class Discriminator(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Discriminator, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            dc5=L.Linear(initial_size * initial_size * int(128 * density), 2,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))
            return self.dc5(h4), h3


class Generator48(chainer.Chain):
    def __init__(self):
        latent_size = 100
        super(Generator48, self).__init__(
            g1=L.Linear(latent_size * 2, 6 * 6 * 128, initialW=Normal(0.02)),
            norm1=L.BatchNormalization(6 * 6 * 128),
            g2=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=Normal(0.02)),
            norm2=L.BatchNormalization(64),
            g3=L.Deconvolution2D(64, 32, 4, stride=2, pad=1, initialW=Normal(0.02)),
            norm3=L.BatchNormalization(32),
            g4=L.Deconvolution2D(32, 1, 4, stride=2, pad=1, initialW=Normal(0.02)),
        )
        self.latent_size = latent_size

    def __call__(self, (z, y), train=True):
        with chainer.using_config('train', train):
            h1 = F.reshape(F.relu(self.norm1(self.g1(z))), (z.data.shape[0], 128, 6, 6))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h3 = F.relu(self.norm3(self.g3(h2)))
            h4 = F.sigmoid(self.g4(h3))
            return h4


class Discriminator48(chainer.Chain):
    def __init__(self):
        super(Discriminator48, self).__init__(
            dc1=L.Convolution2D(1, 32, 4, stride=2, pad=1, initialW=Normal(0.02)),
            norm1=L.BatchNormalization(32),
            dc2=L.Convolution2D(32, 64, 4, stride=2, pad=1, initialW=Normal(0.02)),
            norm2=L.BatchNormalization(64),
            dc3=L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=Normal(0.02)),
            norm3=L.BatchNormalization(128),
            dc4=L.Linear(6 * 6 * 128, 2, initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.norm1(self.dc1(x)))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            return self.dc4(h3)
