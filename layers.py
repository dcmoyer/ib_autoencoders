import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda, merge, Dense
from keras import activations
from keras.initializers import Constant
import numpy as np
import tensorflow as tf 
from random import shuffle

def vae_sample(args):
  # standard reparametrization trick: N(0,1) => N(mu(x), sigma(x))
  z_mean, z_noise = args
  std = 1.0
  if not hasattr(z_mean, '_keras_shape'):
    z_mean = K.variable(z_mean)
  z_score = K.random_normal(shape=(z_mean._keras_shape[1],),
                                  mean=0.,
                                  stddev=std)
  return z_mean + K.exp(z_noise / 2) * z_score
    #return z_mean + z_noise * epsilon

def ido_sample(args):
  # reparametrization trick in log normal space (i.e. multiplicative noise)
  z_mean, z_noise = args
  std = 1.0
  z_score = K.random_normal(shape=(z_mean._keras_shape[1],),
                                  mean=0.,
                                  stddev=std)
    
  return K.exp(K.log(z_mean) + K.exp(z_noise / 2) * z_score)
  #return K.exp(K.log(z_mean) + z_noise * epsilon)

def echo_capacity(args, init = -5, batch = 100):
  print("ARGS (capacity only) : ", args)
  print("init ", init)
  if isinstance(args, list):
    print("args is a list")
    z_mean = args[0] # only one stat argument to echo sample (mean, no variance)
  else:
    print("args not list")
    z_mean = args
  #if isinstance(args, list):
  #  z_mean = args[0] # only one stat argument to echo sample (mean, no variance)
  #else:
  #  z_mean = args
  with tf.variable_scope('encoder_noise', reuse=tf.AUTO_REUSE):
    #latent_shape = z_mean.get_shape().as_list()[1:] if not hasattr(z_mean, '_keras_shape') else z_mean._keras_shape[1:]
    #init = tf.constant(init, shape=latent_shape, dtype=tf.float32)  # Init with very small noise
    cap_param = tf.get_variable("capacity_parameter")#, initializer=init)
    #phi = tf.get_variable('phi', initializer=tf.constant(np.pi, shape=latent_shape, dtype=tf.float32))
    #c = tf.sigmoid(cap_param, name="e_cap")
    return cap_param

def echo_sample(args, init = -5., d_max = 50, batch = 100, multiplicative = False, trainable = False, periodic = False):
  print()
  print("ARGS echo sample : ", args)
  print("init ", init, " dmax ", d_max)
  if isinstance(args, list):
    z_mean = args[0] # only one stat argument to echo sample (mean, no variance)
  else:
    z_mean = args
  print(" Z MEAN ", z_mean)
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
      #assert d_max < batch_size, "d_max < batch_size, integers. Strictly less."
      inds = []
      for i in range(batch_size):
          sub_batch = list(range(batch_size))
          sub_batch.pop(i)
          shuffle(sub_batch)
          # inds.append(list(enumerate([i] + sub_batch[:d_max])))
          inds.append(list(enumerate(sub_batch[:d_max])))
      return inds

  # td: batch = None is problematic: made it a kwarg auto-filled in model...
  #batch = z_mean.get_shape().as_list()[0] if not hasattr(z_mean, '_keras_shape') else K.int_shape(z_mean)[0] #K.cast(K.shape(z_mean)[0], K.floatx()) 
  with tf.variable_scope('encoder_noise', reuse=tf.AUTO_REUSE):
    latent_shape = z_mean.get_shape().as_list()[1:] if not hasattr(z_mean, '_keras_shape') else z_mean._keras_shape[1:]
    init = tf.constant(init, shape=latent_shape, dtype=tf.float32)  # Init with very small noise
    cap_param = tf.get_variable("capacity_parameter", initializer=init, trainable = trainable)
    tf.Session().run(cap_param.initializer)
    phi = tf.get_variable('phi', initializer=tf.constant(np.pi, shape=latent_shape, dtype=tf.float32))
    c = tf.sigmoid(cap_param, name="e_cap")

    print()
    print("ECHO SAMPLE BATCH (size: ", batch, ")")
    print("capacity: ", c.get_shape().as_list())
    print("encoder: ", z_mean.get_shape().as_list())
    

    
    #if batch is not None:
    inds = permute_neighbor_indices(batch, d_max)
    #else:
    #  inds = [list(range(d_max))]
    print("inds : ", len(inds), " , each len ", len(inds[0]))
    inds = tf.constant(inds, dtype=tf.int32)
    if multiplicative:
        normal_encoder = z_mean #tf.log(z_mean + 1e-5) # noise calc done in log space
    else:
        normal_encoder = z_mean

    if periodic:
      c_z_stack = tf.stack([tf.cos(k * phi) * tf.pow(c, k) * normal_encoder for k in range(d_max)])
    else:
      c_z_stack = tf.stack([tf.pow(c, k) * normal_encoder for k in range(d_max)])  # no phase
    
    #noise = tf.gather(c_z_stack, inds, axis = 0)
    print("inds: ", inds)
    print("c_z stack shape ", c_z_stack.shape)
    noise = tf.gather_nd(c_z_stack, inds)
    print("Noise tensor: ", noise.shape)
    # Note : made changes here
    #noise = tf.reduce_sum(noise, axis = 0)
    noise = tf.reduce_sum(noise, axis=1)  # Sums over d_max terms in sum
    noise -= tf.reduce_mean(noise, axis=0)  # Add constant so batch mean is zero
    print("Final noise tensor ", noise.shape, " z mean ", z_mean.shape)

    if multiplicative:
        noisy_encoder = z_mean * tf.exp(c * noise)
    else:
        noisy_encoder = z_mean + c * noise

    return noisy_encoder  

def my_predict(model, data, layer_name, multiple = True):
        func = K.function([model.layers[0].get_input_at(0)],
                        [model.get_layer(layer_name).get_output_at(0)])
        return func([data])[0]

class Beta(Layer):
  def __init__(self, beta = None, trainable = False, **kwargs):
      self.shape = 1
      self.trainable = trainable
      if beta is not None:
        self.beta = beta
      else:
        self.beta = 1.0
   
      super(Beta, self).__init__(**kwargs)

  def build(self, input_shape):
      self.dim = input_shape[1]
      if self.trainable:
        self.betas = self.add_weight(name='beta', 
                                      shape = (self.shape,),
                                      initializer= Constant(value = self.beta),
                                      trainable= True)
      else:
        self.betas = self.add_weight(name='beta', 
                              shape = (self.shape,),
                              initializer= Constant(value = self.beta),
                              trainable= False)
      super(Beta, self).build(input_shape) 

  def call(self, x):
    repeat = 1
    return K.repeat_elements(K.expand_dims(self.betas,1), repeat, -1)

  #not used externally
  def set_beta(self, beta):
      self.set_weights([np.array([beta])])
      

  def get_beta(self):
      return self.get_weights()[0][0]

  def compute_output_shape(self, input_shape):
      return (1, 1)
      #(input_shape[0], self.dim)


class BetaUnTrain(Layer):
  def __init__(self, beta = None, **kwargs):
      self.shape = 1
      if beta is not None:
        self.beta = beta
      else:
        self.beta = 1.0
   
      super(BetaUnTrain, self).__init__(**kwargs)

  def build(self, input_shape):
      self.dim = input_shape[1]
      self.betas = self.add_weight(name='beta', 
                            shape = (self.shape,),
                            initializer= Constant(value = self.beta),
                            trainable= False)
      super(BetaUnTrain, self).build(input_shape) 

  def call(self, x):
    repeat = 1
    return K.repeat_elements(K.expand_dims(self.betas,1), repeat, -1)

  #not used externally
  def set_beta(self, beta):
      self.set_weights([np.array([beta])])

  def get_beta(self):
      return self.get_weights()[0][0]

  def compute_output_shape(self, input_shape):
      return (1, 1)
      #(input_shape[0], self.dim)



