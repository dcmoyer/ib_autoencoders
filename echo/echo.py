""" CorEx with Echo Noise

Code below by:
Greg Ver Steeg (gregv@isi.edu), 2018.
"""

import os
import time
from random import shuffle
import pickle
import numpy as np
import tensorflow as tf  # TF 1.0 or greater
from tensorflow.python import debug as tf_debug
import scipy.misc
import models
import keras.backend as K

class Echo(object):
    """
    Base class with Echo Noise.
    A generic architecture is specified in "architecture". For experiments, I recommend subclassing and
    replacing just the architecture method with something that is parametrized for the experiments you'd like to do.

    Conventions
    ----------
    Code follows sklearn naming/style (e.g. fit(X) to train, transform() to apply model to test data,
    predict() recovers inputs from latent factors.

    Parameters
    ----------
    verbose : int, optional
        Print verbose outputs.

    epochs : int, default=100
        Epochs used in training

    batch_size : int, default=None
        None uses all data in a single batch.

    architecture_args : dictionary of arguments to pass to architecture building routine.

    noise: {"none", "independent", "correlated"} : pick whether to include noise,
                                                   and whether it should be correlated or independent

    noise_type : {'additive', 'multiplicative'}, default='additive'
        Whether to use additive noise (default) or multiplicative noise.

    binary_inputs : bool
        Input variables in range [0,1] treated as binary probability

    d_max : int, default=3
        This controls the quality of the approximation for noise with bounded capacity. Higher d_max should be better
        but two caveats: 1. It's slower to have larger d_max. 2. batch_size has to go up concomitantly. Ideally,
        batch_size > 30 * d_max, mayber even 50 * d_max.
    """

    def __init__(self, epochs=100, batch_size=None, learning_rate=1e-3, verbose=False, debug=False,
                 architecture_args={}, binary_inputs=False,  # Architecture
                 noise='none', noise_type='additive', d_max=-1):  # Noise details
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.verbose = verbose
        self.architecture_args = architecture_args  # Passed to autoencoder specification
        self.noise = noise  # {"none", "independent", "correlated"}
        self.binary_inputs = binary_inputs  # Input variables in range [0,1] treated as binary probability
        self.multiplicative = (noise_type == 'multiplicative' and noise == 'correlated')  # Whether to use multiplicative noise (else additive)
        if self.multiplicative:
            self.architecture_args['activation'] = tf.nn.softplus  # Hard-coded per Achille, Soatto dropout paper
        if d_max < 0:
            d_max = batch_size + d_max
        self.d_max = d_max  # A parameter that controls the quality of the capacity bound. Higher is better, but slower.
        self.c_min = 16. / d_max  # -log 2**-23 / d_max. float32 has 23 bits in mantissa, noise beyond will not matter

        # Logging
        log_root = os.path.join(os.path.expanduser("~"), 'tmp/tensorflow/')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        last_run = max([0] + [int(k) for k in os.listdir(log_root) if k.isdigit()])
        log_dir = os.path.join(log_root, '{0:04d}'.format(last_run + 1))  # Each run log gets a new directory
        os.makedirs(log_dir)
        if verbose:
            print("Visualize logs using: tensorboard --logdir={0}".format(log_dir))
        pickle.dump(self.__dict__, open(os.path.join(log_dir, 'kw_dict.pkl'), 'wb'))
        self.log_dir = log_dir

        # Debugging
        tf.reset_default_graph()  # To avoid naming conflicts in interactive sessions
        self.sess = tf.Session()  # Use this session for running
        if debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Initialize these when we fit on data
        self.n_samples, self.data_shape = 0, (0,)  # Number of samples/variables in input data
        self.encoder, self.decoder, self.input_tensor, self.noisy_encoder = None, None, None, None  # Named layers
        self.loss = None  # Loss
        self.measures = {}  # Store some interesting info measures after running

    def architecture(self, input_tensor, noise='none', reuse=False):
        """Build the encoder/decoder. The input_tensor is the TF node for feeding in data."""
        encoder = models.build_encoder(input_tensor, self.architecture_args, reuse=reuse)
        noisy_encoder = self.build_noise(encoder, noise=noise, reuse=reuse)
        decoder = models.build_decoder(noisy_encoder, self.architecture_args, self.data_shape, reuse=reuse)
        return encoder, noisy_encoder, decoder

    def build_noise(self, encoder, noise='none', reuse=False):
        """Build the (echo) noise. Should work for latent space of arbitrary shape (TODO: test)."""
        with tf.variable_scope('encoder_noise', reuse=reuse):
            latent_shape = encoder.get_shape().as_list()[1:]
            init = tf.constant(-5., shape=latent_shape, dtype=tf.float32)  # Init with very small noise
            cap_param = tf.get_variable("capacity_parameter", initializer=init)
            phi = tf.get_variable('phi', initializer=tf.constant(np.pi, shape=latent_shape, dtype=tf.float32))
            c = tf.sigmoid(cap_param, name="e_cap")  # Parametrization insures the coefficients are in (0, 1)
            print("Capacity Parameter: ", c.shape)
            print(c)
            print("latent shape: ", encoder.get_shape().as_list())
            if noise == 'independent':
                assert False, "Independent noise, NOT IMPLEMENTED"
            elif noise == 'correlated':
                print("Echo noise")
                inds = permute_neighbor_indices(self.batch_size, self.d_max)
                #inds = tf.constant(permute_neighbor_indices(self.batch_size, self.d_max), dtype=tf.int32)
                print("inds ", len(inds), " , each of len ", len(inds[1]))
                inds = tf.constant(inds, dtype=tf.int32)
                if self.multiplicative:
                    normal_encoder = encoder #tf.log(encoder + 1e-5)
                else:
                    normal_encoder = encoder
                c_z_stack = tf.stack([tf.cos(k * phi) * tf.pow(c, k) * normal_encoder for k in range(self.d_max)])
                # c_z_stack = tf.stack([tf.pow(c, k) * normal_encoder for k in range(self.d_max)])  # no phase
                print("c_z_stack size ", c_z_stack.shape)
                noise = tf.gather_nd(c_z_stack, inds)
                noise = tf.reduce_sum(noise, axis=1)  # Sums over d_max terms in sum
                noise -= tf.reduce_mean(noise, axis=0)  # Add constant so batch mean is zero
                if self.multiplicative:
                    noisy_encoder = encoder * tf.exp(c * noise)
                else:
                    noisy_encoder = encoder + c * noise
            else:
                print("No noise (omit compression term)")
                noisy_encoder = tf.identity(encoder)
        return noisy_encoder

    def build_loss(self, input_tensor, encoder, decoder):
        """ Build the computational graph for calculating the loss."""
        with tf.variable_scope('encoder_noise', reuse=True):
            cap_param = tf.get_variable("capacity_parameter")
        with tf.name_scope('loss'):
            n_observed = np.prod(self.data_shape)
            if self.binary_inputs:
                recon_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder, labels=input_tensor)
                recon_loss = tf.reduce_mean(recon_loss, axis=0)
                recon_loss = tf.reduce_sum(recon_loss, name='recon_loss')
            else:
                h_const = tf.constant(0.5 * np.log(2. * np.pi) * n_observed, dtype=tf.float32)
                recon_error = tf.subtract(input_tensor, decoder, name="recon_error")
                mse = tf.reduce_mean(tf.square(recon_error), axis=0, name='mean_error')
                recon_loss = tf.add(h_const, 0.5 * tf.reduce_sum(tf.log(mse + 1e-5)), name='recon_loss')
            if self.noise in ['correlated', 'independent']:
                #capacities = tf.identity(tf.nn.softplus(-cap_param) - np.log(self.c_min), name='capacities')
                capacities = tf.maximum(tf.nn.softplus(- cap_param), self.c_min, name='capacities')
            else:
                capacities = tf.identity(0., name='capacities')
            self.reg = tf.reduce_sum(capacities, name="capacity")
            self.regvar = K.var(capacities)
            loss = tf.add(recon_loss, self.reg, name='total_loss')
        return loss

    def transform(self, x):
        """Transform an array of inputs, x, into the first layer factors or a hierarchy of latent factors."""
        x = load_data(x)
        # TODO: batch
        return self.sess.run(self.encoder, feed_dict={self.input_tensor: x})

    def predict(self, y):
        """Decode latent factors to recover inputs.
           This only predicts the means, use generate to sample(?).
        """
        # TODO: batch
        shape = (y.shape[0],) + self.data_shape
        y = self.sess.run(self.decoder, feed_dict={self.noisy_encoder: y,
                                                   self.input_tensor: np.zeros(shape)})  # Dummy to fix the batch size
        return y

    def fit_transform(self, x):
        """Train and then transform x to latent factors, y."""
        self.fit(x)
        return self.transform(x)

    def fit(self, data, val_data=None):
        """Train. Validation data is optional, only used for logging."""
        self.n_samples = len(data)
        self.data_shape = load_data(data).shape[1:]
        if self.batch_size is None:
            self.batch_size = self.n_samples

        # Build the encoder/decoder
        self.input_tensor = tf.placeholder(tf.float32, shape=(None,) + self.data_shape, name='input')
        self.encoder, self.noisy_encoder, self.decoder = self.architecture(self.input_tensor, noise=self.noise)

        # Build the computational graph for the loss function / objective
        self.loss = self.build_loss(self.input_tensor, self.encoder, self.decoder)  # Losses at each layer

        # Train the model
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # Log quantities for tensorboard
        summary_train, summary_val, writer = self.log(log_dir=self.log_dir, graph=tf.get_default_graph())
        saver = tf.train.Saver()
        with self.sess.as_default():
            tf.global_variables_initializer().run()
            for i in range(self.epochs):  # Outer training loop
                perm = np.random.permutation(self.n_samples)  # random permutation of data for each epoch
                t0 = time.time()
                for offset in range(0, (int(self.n_samples / self.batch_size) * self.batch_size), self.batch_size):  # inner
                    batch_data = load_data(data, perm[offset:(offset + self.batch_size)])
                    result = self.sess.run([train_step, summary_train, self.loss, self.reg, self.regvar],
                                           feed_dict={self.input_tensor: batch_data})
                    summary, loss, reg = result[1], result[2], result[3]
                    regvar = result[4]
                writer.add_summary(summary, i)
                if val_data is not None:
                    assert len(val_data) == self.batch_size, "Must compare with batches of equal size"
                    val_data = load_data(val_data)
                    summary, val_loss = self.sess.run([summary_val, self.loss],
                                                      feed_dict={self.input_tensor: val_data})
                    writer.add_summary(summary, i)
                else:
                    val_loss = np.nan
                if self.verbose:
                    t = time.time()
                    print('{}/{}, Loss:{:0.3f}, Val:{:0.3f}, Echo:{:0.3f}, EchoVar:{:0.3f}, Seconds: {:0.1f}'.
                          format(i, self.epochs, loss, val_loss, reg, regvar, t - t0))
                    if i % 1000 == 999:
                        print('Saving at {} into {}'.format(i, self.log_dir))
                        saver.save(self.sess, os.path.join(self.log_dir, "model_{}.ckpt".format(i)))

        # Denouement
        self.calculate_details()  # Calculate some derived quantities used in analysis
        # We construct a new encoder, decoder for testing... without noise
        self.encoder, self.noisy_encoder, self.decoder = self.architecture(self.input_tensor, noise='none', reuse=True)
        return self

    def _get_parameters(self, weight, coder, layer):
        """Access variables using naming scheme."""
        return tf.get_default_graph().get_tensor_by_name("{}/{}/dense/{}:0".format(coder, layer, weight))

    def log(self, log_dir, graph):
        """Quantities to log in tensorboard."""
        tf.summary.scalar("Loss", self.loss)
        tf.summary.histogram('Zs', self.encoder)
        tf.summary.scalar('capacity', graph.get_tensor_by_name('loss/capacity:0'))
        tf.summary.scalar('recon_loss', graph.get_tensor_by_name('loss/recon_loss:0'))
        tf.summary.histogram('Capacities', graph.get_tensor_by_name('loss/capacities:0'))
        if self.architecture_args.get('type', 'fc') == 'fc':
            for i in range(len(self.architecture_args['layers'])):
                tf.summary.histogram('W_e_{}'.format(i), self._get_parameters('kernel', 'encoder', i))
                tf.summary.histogram('b_e_{}'.format(i), self._get_parameters('bias', 'encoder', i))
                tf.summary.histogram('W_d_{}'.format(i), self._get_parameters('kernel', 'decoder', i))
                tf.summary.histogram('b_d_{}'.format(i), self._get_parameters('bias', 'decoder', i))
            if len(self.data_shape) == 2:
                tf.summary.image('decoder_weights', tf.reshape(
                    self._get_parameters('kernel', 'decoder', 0), (-1,) + self.data_shape + (1,)), max_outputs=50)
                tf.summary.image('encoder_weights', tf.reshape(tf.transpose(
                    self._get_parameters('kernel', 'encoder', 0)), (-1,) + self.data_shape + (1,)), max_outputs=50)
        summary_train = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, graph=graph)
        val_loss = tf.summary.scalar("Val.Loss", self.loss)
        val_recon = tf.summary.scalar('Val.recon_loss', graph.get_tensor_by_name('loss/recon_loss:0'))
        return summary_train, tf.summary.merge([val_loss, val_recon]), writer

    def calculate_details(self):
        """Optionally calculate some derived quantities after training for easier interpretation."""
        graph = tf.get_default_graph()
        self.measures['I(Z_j;X)'] = self.sess.run(graph.get_tensor_by_name('loss/capacities:0'))
        return True


class EchoSup(Echo):
    """
    Supervised information bottleneck using echo noise

    Conventions: Code follows sklearn naming/style (e.g. fit(X, y) to train, transform() to apply model to test data.
    Parameters: see base class
    beta : trade-off between compression and relevance, objective = I(Z;Y) - beta * I(Z;X)
    """

    def __init__(self, beta=1, categorical=True, **kwargs):
        self.beta = beta
        self.categorical = categorical  # Whether the labels are considered categorical or continuous (regression)
        self.labels = None  # Used as placeholder for labels in training
        self.depth = None  # Number of classes for categorical, specified in training
        super(EchoSup, self).__init__(**kwargs)

    def fit(self, data, labels, val_data=None, val_labels=None):
        """Train. Validation data is optional, only used for logging."""
        self.n_samples = len(data)
        self.data_shape = load_data(data).shape[1:]
        if self.batch_size is None:
            self.batch_size = self.n_samples
        if self.categorical:
            self.depth = len(np.unique(labels))

        # Build the encoder/decoder
        self.input_tensor = tf.placeholder(tf.float32, shape=(None,) + self.data_shape, name='input')
        self.labels = tf.placeholder(tf.int32, shape=(None,), name='output')
        self.encoder, self.noisy_encoder, self.decoder = self.architecture(self.input_tensor, noise=self.noise)

        # Build the computational graph for the loss function / objective
        self.loss = self.build_loss(self.input_tensor, self.decoder, self.labels)  # Losses at each layer

        # Train the model
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # Log quantities for tensorboard
        summary_train, summary_val, writer = self.log(log_dir=self.log_dir, graph=tf.get_default_graph())
        saver = tf.train.Saver()
        with self.sess.as_default():
            tf.global_variables_initializer().run()
            for i in range(self.epochs):  # Outer training loop
                perm = np.random.permutation(self.n_samples)  # random permutation of data for each epoch
                t0 = time.time()
                for offset in range(0, ((self.n_samples / self.batch_size) * self.batch_size), self.batch_size):  # inner
                    batch_data = load_data(data, perm[offset:(offset + self.batch_size)])
                    batch_labels = labels[perm[offset:(offset + self.batch_size)]]
                    result = self.sess.run([train_step, summary_train, self.loss],
                                           feed_dict={self.input_tensor: batch_data,
                                                      self.labels: batch_labels})
                    summary, loss = result[1], result[2]
                writer.add_summary(summary, i)
                if val_data is not None:
                    assert len(val_data) == self.batch_size, "Must compare with batches of equal size"
                    val_data = load_data(val_data)
                    summary, val_loss = self.sess.run([summary_val, self.loss],
                                                      feed_dict={self.input_tensor: val_data,
                                                                 self.labels: val_labels})
                    writer.add_summary(summary, i)
                else:
                    val_loss = np.nan
                if self.verbose:
                    t = time.time()
                    print('{}/{}, Loss:{:0.3f}, Val:{:0.3f}, Seconds: {:0.1f}'.
                          format(i, self.epochs, loss, val_loss, t - t0))
                    if i % 1000 == 999:
                        print('Saving at {} into {}'.format(i, self.log_dir))
                        saver.save(self.sess, os.path.join(self.log_dir, "model_{}.ckpt".format(i)))

        # Denouement
        self.calculate_details()  # Calculate some derived quantities used in analysis
        # We construct a new encoder, decoder for testing... without noise
        self.encoder, self.noisy_encoder, self.decoder = self.architecture(self.input_tensor, noise='none', reuse=True)
        return self

    def architecture(self, input_tensor, noise='none', reuse=False):
        """Build the encoder/decoder. The input_tensor is the TF node for feeding in data.
           In the supervised case, the decoder just gives logits to categorical output or outputs regression."""
        encoder = models.build_encoder(input_tensor, self.architecture_args, reuse=reuse)
        noisy_encoder = self.build_noise(encoder, noise=noise, reuse=reuse)
        with tf.variable_scope('decoder', reuse=reuse):
            in_shape = np.prod(noisy_encoder.get_shape().as_list()[1:])
            decoder = tf.reshape(noisy_encoder, [-1, in_shape])
            if self.categorical:
                decoder = tf.layers.dense(decoder, units=self.depth, activation=None, name='dense')
            else:
                decoder = tf.layers.dense(decoder, units=1, activation=None, name='dense')
        return encoder, noisy_encoder, decoder

    def build_loss(self, input_tensor, decoder, labels):
        """ Build the computational graph for calculating the loss."""
        with tf.variable_scope('encoder_noise', reuse=True):
            cap_param = tf.get_variable("capacity_parameter")
        with tf.name_scope('loss'):
            if self.categorical:
                labels = tf.one_hot(labels, self.depth)
                recon_loss = tf.nn.softmax_cross_entropy_with_logits(logits=decoder, labels=labels)  # defaults to last dimension
                recon_loss = tf.reduce_mean(recon_loss, axis=0, name='recon_loss')
            else:
                h_const = tf.constant(0.5 * np.log(2. * np.pi), dtype=tf.float32)
                recon_error = tf.subtract(labels, decoder, name="recon_error")
                mse = tf.reduce_mean(tf.square(recon_error), axis=0, name='mean_error')
                recon_loss = tf.add(h_const, 0.5 * tf.reduce_sum(tf.log(mse + 1e-5)), name='recon_loss')
            if self.noise in ['correlated', 'independent']:
                #capacities = tf.identity(tf.nn.softplus(-cap_param) - np.log(self.c_min), name='capacities')
                capacities = tf.maximum(tf.nn.softplus(- cap_param), self.c_min, name='capacities')
            else:
                capacities = tf.identity(0., name='capacities')
            reg = tf.reduce_sum(capacities, name="capacity")
            loss = tf.add(recon_loss, self.beta * reg, name='total_loss')
        return loss

    def transform(self, x):
        """Transform an array of inputs, x, into the first layer factors or a hierarchy of latent factors."""
        x = load_data(x)
        return self.sess.run(self.encoder, feed_dict={self.input_tensor: x})

    def predict(self, x):
        """Decode latent factors to recover inputs."""
        # TODO: batch
        y = self.sess.run(self.decoder, feed_dict={self.input_tensor: x,
                                                   self.labels: np.zeros(len(x))})  # Dummy to fix the batch size
        return np.argmax(y, axis=1)

    def log(self, log_dir, graph):
        """Quantities to log in tensorboard."""
        tf.summary.scalar("Loss", self.loss)
        tf.summary.histogram('Zs', self.encoder)
        tf.summary.scalar('capacity', graph.get_tensor_by_name('loss/capacity:0'))
        tf.summary.scalar('recon_loss', graph.get_tensor_by_name('loss/recon_loss:0'))
        tf.summary.histogram('Capacities', graph.get_tensor_by_name('loss/capacities:0'))
        if self.architecture_args.get('type', 'fc') == 'fc':
            for i in range(len(self.architecture_args['layers'])):
                tf.summary.histogram('W_e_{}'.format(i), self._get_parameters('kernel', 'encoder', i))
                tf.summary.histogram('b_e_{}'.format(i), self._get_parameters('bias', 'encoder', i))
            if len(self.data_shape) == 2:
                tf.summary.image('encoder_weights', tf.reshape(tf.transpose(
                    self._get_parameters('kernel', 'encoder', 0)), (-1,) + self.data_shape + (1,)), max_outputs=50)
        summary_train = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, graph=graph)
        val_loss = tf.summary.scalar("Val.Loss", self.loss)
        val_recon = tf.summary.scalar('Val.recon_loss', graph.get_tensor_by_name('loss/recon_loss:0'))
        return summary_train, tf.summary.merge([val_loss, val_recon]), writer


def load(ckpt_file, data_shape):
    directory = os.path.dirname(ckpt_file)
    kw = pickle.load(open(os.path.join(directory, 'kw_dict.pkl')))
    if kw.pop('multiplicative'):
        kw['noise_type'] = 'multiplicative'
    else:
        kw['noise_type'] = 'additive'
    cls = Echo(**kw)
    cls.data_shape = data_shape
    cls.input_tensor = tf.placeholder(tf.float32, shape=(None,) + data_shape, name='input')
    cls.encoder, cls.noisy_encoder, cls.decoder = cls.architecture(cls.input_tensor, noise='none')
    cls.loss = cls.build_loss(cls.input_tensor, cls.encoder, cls.decoder)  # Losses at each layer
    tf.train.Saver().restore(cls.sess, ckpt_file)
    cls.calculate_details()  # Calculate some derived quantities used in analysis
    return cls


def permute_neighbor_indices(batch_size, d_max=-1):
    """Produce an index tensor that gives a permuted matrix of other samples in batch, per sample.
    Parameters
    ----------
    batch_size : int
        Number of samples in the batch.
    d_max : int
        The number of blocks, or the number of samples to generate per sample.
    """
    if d_max < 0:
        d_max = batch_size + d_max
    assert d_max < batch_size, "d_max < batch_size, integers. Strictly less."
    inds = []
    for i in range(batch_size):
        sub_batch = list(range(batch_size))
        sub_batch.pop(i)
        shuffle(sub_batch)
        # inds.append(list(enumerate([i] + sub_batch[:d_max])))
        inds.append(list(enumerate(sub_batch[:d_max])))
    return inds


def load_data(data, indices=None):
    """If data is list of filenames, load images into data batch.
        Otherwise if it is raw data, pass it through, selecting the batch.
    """
    if indices is None:
        indices = range(len(data))
    if type(data) is np.ndarray:
        return data[indices]
    else:
        x = []
        for i in indices:
            path = data[i]
            x.append(scipy.misc.imread(path).astype(np.float))
        return np.array(x).astype(np.float32)
