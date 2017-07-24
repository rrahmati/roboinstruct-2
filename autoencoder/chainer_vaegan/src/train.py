import argparse
import numpy as np
import io, math
import os, sys, time
from PIL import Image
import cPickle as pickle
import thread
import chainer
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import net, rnn_net
import copy
import signal

parser = argparse.ArgumentParser(description='VAE and DCGAN trainer')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default='../model/', type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', default='../model/', type=str,
                    help='output model file path without extension')
parser.add_argument('--iter', default=1000, type=int,
                    help='number of iteration')
parser.add_argument('--out_image_dir', default='../image', type=str,
                    help='output directory to output images')
parser.add_argument('--size', '-s', default=128, type=int, choices=[48, 64, 80, 96, 112, 128],
                    help='image size')
args = parser.parse_args()

cost_mode = 'GAN' # VAE , GAN, BEGAN ** BEGAN not working!
latent_size = 256
BATCH_SIZE = 100
seq_length = 1
gpus_to_use = [0] # You can use multiple GPUs by putting their indices in an array. For instance: [0,1,2,3] for four GPUs
num_gpus = len(gpus_to_use)
main_gpu = gpus_to_use[0]
max_seq_length = 5
kt = 0
lambda_k = 0.001
gamma = .5
train_LSTM = True
train_LSTM_using_cached_features = False
train_lstm_prob = .5
train_dis = True
image_save_interval = 200000
model_save_interval = image_save_interval
out_image_row_num = 7
out_image_col_num = 14
if train_LSTM:
    BATCH_SIZE /= seq_length
normer = args.size * args.size * 3 * 60
image_path = ['../../../trajectories/al5d']
np.random.seed( 1241 )
image_size = args.size
enc_model = [net.Encoder(density=8, size=image_size, latent_size=latent_size)]
gen_model = [net.Generator(density=8, size=image_size, latent_size=latent_size)]
dis_model = [net.Discriminator(density=8, size=image_size)]
for i in range(num_gpus-1):
    enc_model.append(copy.deepcopy(enc_model[0]))
    gen_model.append(copy.deepcopy(gen_model[0]))
    dis_model.append(copy.deepcopy(dis_model[0]))

enc_dis_model = net.Encoder(density=8, size=image_size, latent_size=latent_size)
gen_dis_model = net.Generator(density=8, size=image_size, latent_size=latent_size)
rnn_model = rnn_net.MDN_RNN(IN_DIM=latent_size+5, HIDDEN_DIM=300, OUT_DIM=6, NUM_MIXTURE=40)

optimizer_enc = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_enc.setup(enc_model[0])
optimizer_enc.add_hook(chainer.optimizer.WeightDecay(0.00001))
optimizer_gen = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_gen.setup(gen_model[0])
optimizer_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))

optimizer_enc_dis = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_enc_dis.setup(enc_dis_model)
optimizer_enc_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))
optimizer_gen_dis = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_gen_dis.setup(gen_dis_model)
optimizer_gen_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

optimizer_dis = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_dis.setup(dis_model[0])
optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))
optimizer_rnn = optimizers.Adam(alpha=0.0001, beta1=0.5)
optimizer_rnn.setup(rnn_model)
optimizer_rnn.add_hook(chainer.optimizer.WeightDecay(0.00001))
xp = cuda.cupy

def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

for i in range(num_gpus):
    enc_model[i].to_gpu(gpus_to_use[i])
    gen_model[i].to_gpu(gpus_to_use[i])
    dis_model[i].to_gpu(gpus_to_use[i])
if cost_mode == 'BEGAN':
    enc_dis_model.to_gpu(main_gpu)
    gen_dis_model.to_gpu(main_gpu)
rnn_model.to_gpu(main_gpu)
if args.input != None:
    try:
        if __name__ == '__main__':
            file_path = os.path.join(os.path.dirname( __file__ ), '..', 'model/')
        else:
            file_path = os.path.join(os.path.dirname( __file__ ), '..', 'model-saved/')
        serializers.load_hdf5(file_path + 'enc.model', enc_model[0])
        serializers.load_hdf5(file_path + 'enc.state', optimizer_enc)
        serializers.load_hdf5(file_path + 'gen.model', gen_model[0])
        serializers.load_hdf5(file_path + 'gen.state', optimizer_gen)
        if cost_mode == 'BEGAN':
            serializers.load_hdf5(file_path + 'enc_dis.model', enc_dis_model)
            serializers.load_hdf5(file_path + 'enc_dis.state', optimizer_enc_dis)
            serializers.load_hdf5(file_path + 'gen_dis.model', gen_dis_model)
            serializers.load_hdf5(file_path + 'gen_dis.state', optimizer_gen_dis)

        serializers.load_hdf5(file_path + 'dis.model', dis_model[0])
        serializers.load_hdf5(file_path + 'dis.state', optimizer_dis)
        for g in range(1, num_gpus):
            enc_model[g].copyparams(enc_model[0])
            gen_model[g].copyparams(gen_model[0])
            dis_model[g].copyparams(dis_model[0])
        if train_LSTM:
            serializers.load_hdf5(file_path + 'rnn.model', rnn_model)
            serializers.load_hdf5(file_path + 'rnn.state', optimizer_rnn)
        print 'Model loaded. '
    except:
        print 'cannot load model from {}'.format(file_path)
        # if __name__ != '__main__':
        #     sys.exit(0)

if args.out_image_dir != None:
    if not os.path.exists(args.out_image_dir):
        try:
            os.mkdir(args.out_image_dir)
        except:
            print 'cannot make directory {}'.format(args.out_image_dir)
            exit()
    elif not os.path.isdir(args.out_image_dir):
        print 'file path {} exists but is not directory'.format(args.out_image_dir)
        exit()
def read_images(indices):
    if train_LSTM:
        images = []
        for i in indices:
            image = img_data[i:i+seq_length]
            images.append(image)
        images = np.array(images).swapaxes(0, 1)
        images = images.reshape(len(indices)*seq_length, 3, image_size, image_size)
    else:
        images = []
        for i in indices:
            image = Image.open(image_files[i])
            image = image.resize((args.size, args.size), Image.ANTIALIAS)
            image = image.convert('RGB')
            image = np.array(image)
            image = image.transpose((2, 0, 1))
            image = image[:, :, ::-1].copy()
            images.append(image)
    return images

def read_rnn_data(indices):
    batch_in = []
    batch_out = []
    for i in indices:
        seq_in = data_in[i:i+seq_length]
        seq_out = data_out[i:i+seq_length]
        batch_in.append(seq_in)
        batch_out.append(seq_out)
    batch_in = np.array(batch_in).swapaxes(0, 1)
    batch_out = np.array(batch_out).swapaxes(0, 1)
    return (batch_in, batch_out)

if __name__ == '__main__':
    if train_LSTM:
        img_data = np.load('../../../img_data.npy', mmap_mode='r')
        if train_LSTM_using_cached_features:
            data_in = np.load('../../../data_in_ae.npy')
        else:
            data_in = np.load('../../../data_in.npy')
        data_out = np.load('../../../data_out.npy')[:,:-1]
        print len(img_data), len(data_in), len(data_out)
        n_images = len(img_data)
    else:
        image_files = []
        for image_p in image_path:
            for dirpath, dirnames, filenames in os.walk(image_p):
                if 'camera-1' in dirpath:
                    for filename in [f for f in filenames if f.endswith(".jpg")]:
                        image_files.append(os.path.join(dirpath, filename))
        n_images = len(image_files)
    print 'all images: ',n_images
    indices = np.arange( int(n_images) )
    train_indices = indices[:int(n_images*.70)]
    test_indices = indices[int(n_images*.70):]
    np.random.shuffle( train_indices )
    np.random.shuffle( test_indices )
    test_batch = np.asarray(read_images(test_indices[ 0: out_image_row_num * out_image_col_num / seq_length + seq_length])).astype(np.float32)
    test_batch = test_batch / 127.5 - 1
    x_size = len(train_indices)
    next_batch = np.asarray(read_images(train_indices[ 0 : BATCH_SIZE ])).astype(np.float32)
    next_batch = next_batch / 127.5 - 1
    print 'train images: ', x_size
    print 'Mode: ', cost_mode
    loading_next_batch = False

def get_latent_size():
    return latent_size

def load_next_batch(i):
    global next_batch
    global loading_next_batch
    # next_batch = np.asarray(train_images[train_indices[ i : i+BATCH_SIZE ]]).astype(np.float32)
    next_batch = np.asarray(read_images(train_indices[ i : i+BATCH_SIZE/seq_length ])).astype(np.float32)
    next_batch = next_batch / 127.5 - 1
    loading_next_batch = False

def encode(img_batch):
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        cuda.get_device(main_gpu).use()
        x_in = xp.asarray(img_batch)
        z, mean, var, _ = enc_model[0](Variable(x_in), train=False)
        return (cuda.to_cpu(z.data), cuda.to_cpu(mean.data), cuda.to_cpu(var.data))

def decode(z):
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        cuda.get_device(main_gpu).use()
        z_in = xp.asarray(z)
        x = gen_model[0](Variable(z_in), train=False)
        return ((cuda.to_cpu(x.data) + 1) * 128).clip(0, 255).astype(np.uint8)

def compute_mse_loss(enc, gen, x, xp):
    x = x
    ln_var = math.log(.3 ** 2)
    noise = F.gaussian(Variable(xp.zeros_like(x.data)), Variable(xp.full_like(x.data, ln_var)))
    x = x + noise
    z, mean, _, h3 = enc(x)
    _x = gen(mean)
    return (F.mean_absolute_error(x, _x), _x)

def train_one(enc, gen, dis, rnn, optimizer_enc, optimizer_gen, optimizer_dis, iteration):
    global seq_length
    global kt
    global train_dis
    if train_LSTM_using_cached_features:
        rnn_loss = 0
        rnn_in, rnn_out = read_rnn_data(train_indices[ iteration : iteration+BATCH_SIZE ])
        rnn_in = Variable(xp.asarray(rnn_in, dtype=xp.float32))
        rnn_out = Variable(xp.asarray(rnn_out, dtype=xp.float32))
        rnn_model.reset_state()
        rnn_model.cleargrads()
        for i in range(seq_length):
            rnn_loss += rnn_model(rnn_in[i], rnn_out[i])
        rnn_loss.backward()
        optimizer_rnn.update()
        return (float(0), float(0), float(0), float(0), float(rnn_loss.data))
    if np.random.multinomial(1, [train_lstm_prob, 1-train_lstm_prob])[0] == 1:
        train_lstm_this_iteration = True
        seq_length = max_seq_length
    else:
        train_lstm_this_iteration = False
        seq_length = 1
        # global loading_next_batch
        # while loading_next_batch:
        #     time.sleep(.001)
        # img_batch = next_batch
        # loading_next_batch = True
        # thread.start_new_thread( load_next_batch, (iteration,) )
    img_batch = load_next_batch(iteration)
    img_batch = next_batch
    loss_enc = 0
    rnn_loss = 0
    if train_lstm_this_iteration:
        cuda.get_device(main_gpu).use()
        x_in = xp.asarray(img_batch)
        z0, mean, var, _ = enc[0](Variable(x_in))
        z0_reshaped = F.reshape(z0, (seq_length, z0.shape[0]/seq_length, latent_size))
        rnn_in, rnn_out = read_rnn_data(train_indices[ iteration : iteration+BATCH_SIZE/seq_length ])
        if rnn_in.shape[1] != z0_reshaped.shape[1]:
            return (float(0), float(0), float(0), float(0), float(0))
        rnn_in = Variable(xp.asarray(rnn_in, dtype=xp.float32))
        rnn_out = Variable(xp.asarray(rnn_out, dtype=xp.float32))
        rnn_model.reset_state()
        rnn_model.cleargrads()
        for i in range(seq_length):
            rnn_loss += rnn_model(F.concat((z0_reshaped[i], rnn_in[i]), axis=1), rnn_out[i])
        rnn_loss.backward()
        rnn_loss /= seq_length
        optimizer_rnn.update()
        loss_enc += rnn_loss / 100.
        loss_enc += F.gaussian_kl_divergence(mean, var) / (normer) / 10.
        loss_enc /= 4.
        optimizer_enc.zero_grads()
        loss_enc.backward()
        optimizer_enc.update()
        if train_lstm_this_iteration:
            return (float(loss_enc.data), float(0), float(0), float(0), float(rnn_loss.data))
    if cost_mode == 'VAE':
        loss_enc += F.gaussian_kl_divergence(mean, var) / (normer)
        loss_enc += loss_reconstruction
        loss_gen = loss_reconstruction

        optimizer_enc.zero_grads()
        loss_enc.backward()
        optimizer_enc.update()

        optimizer_gen.zero_grads()
        loss_gen.backward()
        optimizer_gen.update()

        return (float(loss_enc.data), float(loss_gen.data), 0, float(loss_reconstruction.data), 0.)
    elif cost_mode == 'GAN':
        for i, g in enumerate(gpus_to_use):
            cuda.get_device(g).use()
            img_batch_for_gpu = img_batch[i*BATCH_SIZE//num_gpus:(i+1)*BATCH_SIZE//num_gpus]
            gpu_batch_size = len(img_batch_for_gpu)
            # encode
            x_in = cuda.to_gpu(img_batch_for_gpu, g)
            z0, mean, var, _ = enc[i](Variable(x_in))
            x0 = gen[i](z0)
            loss_reconstruction = F.mean_squared_error(x0, x_in)
            y0, l0 = dis[i](x0)
            l_dis_rec = F.softmax_cross_entropy(y0, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size).astype(np.int32),g))) / gpu_batch_size
            z1 = Variable(cuda.to_gpu(xp.random.normal(0, 1, (gpu_batch_size, latent_size), dtype=np.float32),g))
            x1 = gen[i](z1)
            y1, l1 = dis[i](x1)
            l_prior = F.gaussian_kl_divergence(mean, var) / (normer)
            l_dis_fake = F.softmax_cross_entropy(y1, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size).astype(np.int32),g))) / gpu_batch_size
            # # train discriminator
            y2, l2 = dis[i](Variable(x_in))
            l_dis_real = F.softmax_cross_entropy(y2, Variable(cuda.to_gpu(xp.ones(gpu_batch_size).astype(np.int32),g))) / gpu_batch_size
            l_feature_similarity = F.mean_squared_error(l0, l2) #* l2.data.shape[2] * l2.data.shape[3]

            l_dis_sum = (l_dis_real + l_dis_fake + l_dis_rec ) / 3
            loss_enc = l_prior + l_feature_similarity
            loss_gen = l_feature_similarity - l_dis_sum
            loss_dis = l_dis_sum

            enc_model[i].cleargrads()
            loss_enc.backward()

            gen_model[i].cleargrads()
            loss_gen.backward()

            if train_dis:
                dis_model[i].cleargrads()
                loss_dis.backward()
        for i in range(1, num_gpus):
            enc_model[0].addgrads(enc_model[i])
            gen_model[0].addgrads(gen_model[i])
            dis_model[0].addgrads(dis_model[i])
        optimizer_enc.update()
        optimizer_gen.update()
        if train_dis:
            optimizer_dis.update()
        for i in range(1, num_gpus):
            enc_model[i].copyparams(enc_model[0])
            gen_model[i].copyparams(gen_model[0])
            dis_model[i].copyparams(dis_model[0])
        train_dis = float(loss_dis.data) > 0.0001
        return (float(loss_enc.data), float(loss_gen.data), float(loss_dis.data), float(loss_reconstruction.data), .0)

    elif cost_mode == 'BEGAN': # not working!
        # l_dis_rec, l0 = compute_mse_loss(enc_dis_model, gen_dis_model, x0, xp)
        z1 = Variable(xp.random.uniform(-1, 1, (batch_size, latent_size), dtype=np.float32))
        x1 = gen(z1)
        l_dis_fake, l1 = compute_mse_loss(enc_dis_model, gen_dis_model, x1, xp)
        l_prior = F.gaussian_kl_divergence(mean, var) / (normer)
        l_dis_real, l2 = compute_mse_loss(enc_dis_model, gen_dis_model, Variable(xp.asarray(img_batch)), xp)

        l_feature_similarity = 0#F.mean_absolute_error(l0, l2)

        l_fake = l_dis_fake #(l_dis_rec + l_dis_fake)/2
        loss_dis = l_dis_real - kt * l_fake
        loss_gen =  l_fake + l_feature_similarity
        loss_enc += l_prior + l_feature_similarity

        optimizer_enc.zero_grads()
        loss_enc.backward()
        optimizer_enc.update()

        optimizer_gen.zero_grads()
        loss_gen.backward()
        optimizer_gen.update()

        optimizer_enc_dis.zero_grads()
        optimizer_gen_dis.zero_grads()
        loss_dis.backward()
        optimizer_enc_dis.update()
        optimizer_gen_dis.update()

        # update control parameters
        kt += lambda_k * (gamma * float(l_dis_real.data) - float(l_fake.data))
        kt = max(0, min(1, kt))
        M = float(l_dis_real.data) + abs(gamma * float(l_dis_real.data) - float(l_fake.data))
        # sum_M += M

        return (float(loss_enc.data), float(loss_gen.data), float(loss_dis.data), float(loss_reconstruction.data), .0)

def train(enc, gen, dis, optimizer_enc, optimizer_gen, optimizer_dis, epoch_num, out_image_dir=None):
    z_out_image =  Variable(xp.random.uniform(-1, 1, (out_image_row_num * out_image_col_num, latent_size)).astype(np.float32))
    for epoch in xrange(1, epoch_num + 1):
        start_time = time.time()
        sum_loss_enc = sum_loss_gen = sum_loss_dis = sum_loss_rnn = 0
        np.random.shuffle( train_indices )
        for i in xrange(0, x_size - max_seq_length * BATCH_SIZE , BATCH_SIZE):
            batch_start_time = time.time()
            loss_enc, loss_gen, loss_dis, loss_rec, loss_rnn = train_one(enc, gen, dis, rnn_model, optimizer_enc, optimizer_gen, optimizer_dis, i)
            sum_loss_enc += loss_enc * BATCH_SIZE
            sum_loss_gen += loss_gen * BATCH_SIZE
            sum_loss_dis += loss_dis * BATCH_SIZE
            sum_loss_rnn += loss_rnn * BATCH_SIZE
            if i % image_save_interval == 0:
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    print ''
                    print '{} {} {} {}'.format(sum_loss_enc / (image_save_interval), sum_loss_gen / (image_save_interval), sum_loss_dis / (image_save_interval), sum_loss_rnn / (image_save_interval))
                    if out_image_dir != None:
                        cuda.get_device(main_gpu).use()
                        z, m, v, _ = enc[0](Variable(cuda.to_gpu(test_batch,main_gpu)), train=False)
                        z = m
                        data = gen[0](z, train=False).data
                        test_rec_loss = F.squared_difference(data, xp.asarray(test_batch))
                        test_rec_loss = float(F.sum(test_rec_loss).data) / (normer)
                        image = ((cuda.to_cpu(data) + 1) * 128).clip(0, 255).astype(np.uint8)
                        image = image[:out_image_row_num*out_image_col_num]
                        image = image.reshape((out_image_row_num, out_image_col_num, 3, image_size, image_size)).transpose((0, 3, 1, 4, 2)).reshape((out_image_row_num * image_size, out_image_col_num * image_size, 3))
                        Image.fromarray(image).save('{0}/{1:03d}_{2:07d}.png'.format(out_image_dir, epoch, i))
                        if i == 0:
                            org_image = ((test_batch + 1) * 128).clip(0, 255).astype(np.uint8)
                            org_image = org_image[:out_image_row_num*out_image_col_num]
                            org_image = org_image.reshape((out_image_row_num, out_image_col_num, 3, image_size, image_size)).transpose((0, 3, 1, 4, 2)).reshape((out_image_row_num * image_size, out_image_col_num * image_size, 3))
                            Image.fromarray(org_image).save('{0}/org.png'.format(out_image_dir, epoch, i))
                        sum_loss_enc = sum_loss_gen = sum_loss_dis = sum_loss_rnn = 0
            if i % model_save_interval == 0:
                serializers.save_hdf5('{0}enc.model'.format(args.output), enc[0])
                serializers.save_hdf5('{0}enc.state'.format(args.output), optimizer_enc)
                serializers.save_hdf5('{0}gen.model'.format(args.output), gen[0])
                serializers.save_hdf5('{0}gen.state'.format(args.output), optimizer_gen)
                if cost_mode == 'BEGAN':
                    serializers.save_hdf5('{0}enc_dis.model'.format(args.output), enc_dis_model)
                    serializers.save_hdf5('{0}enc_dis.state'.format(args.output), optimizer_enc_dis)
                    serializers.save_hdf5('{0}gen_dis.model'.format(args.output), gen_dis_model)
                    serializers.save_hdf5('{0}gen_dis.state'.format(args.output), optimizer_gen_dis)

                serializers.save_hdf5('{0}dis.model'.format(args.output), dis[0])
                serializers.save_hdf5('{0}dis.state'.format(args.output), optimizer_dis)
                serializers.save_hdf5('{0}rnn.model'.format(args.output), rnn_model)
                serializers.save_hdf5('{0}rnn.state'.format(args.output), optimizer_rnn)
            sys.stdout.write('\r' + str(i/BATCH_SIZE) + '/' + str(x_size/BATCH_SIZE) + ' time: {0:0.2f} errors: {1:0.4f} {2:0.4f} {3:0.8f} {4:0.4f} {5:0.4f} {6:0.4f}'.format(time.time() - batch_start_time, loss_enc, loss_gen, loss_dis, loss_rnn, loss_rec, test_rec_loss))
            sys.stdout.flush()
        print '-----------------------------------------'
        print 'epoch: {} done'.format(epoch)
        print 'time: {}'.format(time.time() - start_time)

if __name__ == '__main__':
    train(enc_model, gen_model, dis_model, optimizer_enc, optimizer_gen, optimizer_dis, args.iter, out_image_dir=args.out_image_dir)
