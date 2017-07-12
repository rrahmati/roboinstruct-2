import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class Encoder(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert(size % 16 == 0)
        initial_size = size / 16
        super(Encoder, self).__init__(
            dc1   = L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            dc2   = L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * density)),
            norm2 = L.BatchNormalization(int(32 * density)),
            dc3   = L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm3 = L.BatchNormalization(int(64 * density)),
            dc4   = L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm4 = L.BatchNormalization(int(128 * density)),
            mean  = L.Linear(initial_size * initial_size * int(128 * density), latent_size, wscale=0.02 * math.sqrt(initial_size * initial_size * 128 * density)),
            var   = L.Linear(initial_size * initial_size * int(128 * density), latent_size, wscale=0.02 * math.sqrt(initial_size * initial_size * 128 * density)),
        )

    def __call__(self, x, train=True):
        xp = cuda.get_array_module(x.data)
        h1 = F.leaky_relu(self.dc1(x))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        mean = self.mean(h4)
        var  = self.var(h4)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.clip(F.exp(var), .001, 100.) * Variable(rand, volatile=not train)
        # z  = mean + F.exp(var) * Variable(rand, volatile=not train)
        return (z, mean, var, h3)

class Generator(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert(size % 16 == 0)
        initial_size = size / 16
        super(Generator, self).__init__(
            g1    = L.Linear(latent_size, initial_size * initial_size * int(128 * density), wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(initial_size * initial_size * int(128 * density)),
            g2    = L.Deconvolution2D(int(128 * density), int(64 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm2 = L.BatchNormalization(int(64 * density)),
            g3    = L.Deconvolution2D(int(64 * density), int(32 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm3 = L.BatchNormalization(int(32 * density)),
            g4    = L.Deconvolution2D(int(32 * density), int(16 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm4 = L.BatchNormalization(int(16 * density)),
            g5    = L.Deconvolution2D(int(16 * density), channel, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * density)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3), test=not train))
        return F.tanh(self.g5(h4))

class Encoder_deep(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert(size % 16 == 0)
        initial_size = size / 16
        super(Encoder_deep, self).__init__(
            dc1   = L.Convolution2D(channel, 16 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            dc2   = L.Convolution2D(16 * density, 32 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * density)),
            norm2 = L.BatchNormalization(32 * density),
            dc3   = L.Convolution2D(32 * density, 64 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm3 = L.BatchNormalization(64 * density),
            dc4   = L.Convolution2D(64 * density, 128 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm4 = L.BatchNormalization(128 * density),

            dc1_   = L.Convolution2D(16 * density, 16 * density, 3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 16 * density)),
            dc2_   = L.Convolution2D(32 * density, 32 * density, 3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 32 * density)),
            norm2_ = L.BatchNormalization(32 * density),
            dc3_   = L.Convolution2D(64 * density, 64 * density, 3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 64 * density)),
            norm3_ = L.BatchNormalization(64 * density),
            dc4_   = L.Convolution2D(128 * density, 128 * density, 3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 128 * density)),
            norm4_ = L.BatchNormalization(128 * density),

            mean  = L.Linear(initial_size * initial_size * 128 * density, latent_size, wscale=0.02 * math.sqrt(initial_size * initial_size * 128 * density)),
            var   = L.Linear(initial_size * initial_size * 128 * density, latent_size, wscale=0.02 * math.sqrt(initial_size * initial_size * 128 * density)),
        )

    def __call__(self, x, train=True):
        xp = cuda.get_array_module(x.data)
        h1 = F.leaky_relu(self.dc1(x))
        h1_ = F.leaky_relu(self.dc1_(h1))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1_), test=not train))
        h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2_), test=not train))
        h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3), test=not train))
        h4 = F.leaky_relu(self.norm4_(self.dc4(h3_), test=not train))
        h4_ = F.leaky_relu(self.norm4_(self.dc4_(h4), test=not train))
        mean = self.mean(h4_)
        var  = self.var(h4_)
        rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
        z  = mean + F.exp(var) * Variable(rand, volatile=not train)
        return (z, mean, var)

class Generator_deep(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        assert(size % 16 == 0)
        initial_size = size / 16
        super(Generator_deep, self).__init__(
            g1    = L.Linear(latent_size, initial_size * initial_size * 128 * density, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(initial_size * initial_size * 128 * density),
            g2    = L.Deconvolution2D(128 * density, 64 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128 * density)),
            norm2 = L.BatchNormalization(64 * density),
            g3    = L.Deconvolution2D(64 * density, 32 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm3 = L.BatchNormalization(32 * density),
            g4    = L.Deconvolution2D(32 * density, 16 * density, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm4 = L.BatchNormalization(16 * density),
            g5    = L.Deconvolution2D(16 * density, channel, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * density)),

            g2_    = L.Deconvolution2D(64 * density, 64 * density, 3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 128 * density)),
            norm2_ = L.BatchNormalization(64 * density),
            g3_    = L.Deconvolution2D(32 * density, 32 * density, 3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 64 * density)),
            norm3_ = L.BatchNormalization(32 * density),
            g4_    = L.Deconvolution2D(16 * density, 16 * density, 3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 32 * density)),
            norm4_ = L.BatchNormalization(16 * density),
            g5_    = L.Deconvolution2D(channel, channel, 3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 16 * density)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 128 * self.density, self.initial_size, self.initial_size))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h2_ = F.relu(self.norm2_(self.g2_(h2), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2_), test=not train))
        h3_ = F.relu(self.norm3_(self.g3_(h3), test=not train))
        h4 = F.relu(self.norm4(self.g4(h3_), test=not train))
        h4_ = F.relu(self.norm4_(self.g4_(h4), test=not train))
        return F.tanh(self.g5(h4_))

class Discriminator(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3):
        assert(size % 16 == 0)
        initial_size = size / 16
        super(Discriminator, self).__init__(
            dc1   = L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * channel * density)),
            dc2   = L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 16 * density)),
            norm2 = L.BatchNormalization(int(32 * density)),
            dc3   = L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32 * density)),
            norm3 = L.BatchNormalization(int(64 * density)),
            dc4   = L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64 * density)),
            norm4 = L.BatchNormalization(int(128 * density)),
            dc5   = L.Linear(initial_size * initial_size * int(128 * density), 2, wscale=0.02 * math.sqrt(initial_size * initial_size * 128 * density)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.dc1(x))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        h4 = F.leaky_relu(self.norm4(self.dc4(h3), test=not train))
        return (self.dc5(h4), h3)

class Generator48(chainer.Chain):
    def __init__(self):
        latent_size = 100
        super(Generator48, self).__init__(
            g1    = L.Linear(latent_size * 2, 6 * 6 * 128, wscale=0.02 * math.sqrt(latent_size)),
            norm1 = L.BatchNormalization(6 * 6 * 128),
            g2    = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            norm2 = L.BatchNormalization(64),
            g3    = L.Deconvolution2D(64, 32, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm3 = L.BatchNormalization(32),
            g4    = L.Deconvolution2D(32, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32)),
        )
        self.latent_size = latent_size

    def __call__(self, (z, y), train=True):
        h1 = F.reshape(F.relu(self.norm1(self.g1(z), test=not train)), (z.data.shape[0], 128, 6, 6))
        h2 = F.relu(self.norm2(self.g2(h1), test=not train))
        h3 = F.relu(self.norm3(self.g3(h2), test=not train))
        h4 = F.sigmoid(self.g4(h3))
        return h4

class Discriminator48(chainer.Chain):
    def __init__(self):
        super(Discriminator48, self).__init__(
            dc1   = L.Convolution2D(1, 32, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4)),
            norm1 = L.BatchNormalization(32),
            dc2   = L.Convolution2D(32, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32)),
            norm2 = L.BatchNormalization(64),
            dc3   = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            norm3 = L.BatchNormalization(128),
            dc4   = L.Linear(6 * 6 * 128, 2, wscale=0.02 * math.sqrt(6 * 6 * 128)),
        )

    def __call__(self, x, train=True):
        h1 = F.leaky_relu(self.norm1(self.dc1(x), test=not train))
        h2 = F.leaky_relu(self.norm2(self.dc2(h1), test=not train))
        h3 = F.leaky_relu(self.norm3(self.dc3(h2), test=not train))
        return self.dc4(h3)
