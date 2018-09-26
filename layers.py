import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda, merge, Dense, Flatten, Reshape
from keras import activations
import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.constraints as constraints
from keras.initializers import Constant
import numpy as np
import tensorflow as tf 
from random import shuffle, randint
import importlib 
import itertools
from collections import defaultdict

tfd = tf.contrib.distributions
tfb = tfd.bijectors

def vae_sample(inputs, std = 1.0):
  # standard reparametrization trick: N(0,1) => N(mu(x), sigma(x))
  z_mean, z_noise = inputs
  if not hasattr(z_mean, '_keras_shape'):
    z_mean = K.variable(z_mean)
  z_score = K.random_normal(shape=(z_mean._keras_shape[-1],),
                                  mean=0.,
                                  stddev=std)
  return z_mean + K.exp(z_noise / 2) * z_score
    #return z_mean + z_noise * epsilon

def ido_sample(inputs):
  # reparametrization trick in log normal space (i.e. multiplicative noise)
  z_mean, z_noise = inputs
  std = 1.0
  z_score = K.random_normal(shape=(z_mean._keras_shape[-1],),
                                  mean=0.,
                                  stddev=std)
    
  return K.exp(z_mean + K.exp(z_noise / 2) * z_score)
  #return K.exp(K.log(z_mean) + K.exp(z_noise / 2) * z_score)
  #return K.exp(K.log(z_mean) + z_noise * epsilon)
def inverse_vae_sample(inputs, std = 1.0):
  # standard reparametrization trick: N(0,1) => N(mu(x), sigma(x))
  target, z_mean, z_noise = inputs
  if not hasattr(z_mean, '_keras_shape'):
    z_mean = K.variable(z_mean)
 
  z_score = (target-z_mean)*K.exp(z_noise)**(-.5)
  return z_score #z_mean + K.exp(z_noise / 2) * z_score

def inverse_flow(inputs):
  pass


def constant_layer(inputs, variance = 1, batch = 200):
# needed to pass BIR variance to loss function
  const = tf.constant(variance, shape = (batch,), dtype = tf.float32)#, shape = tf.shape(z_mean))
  return K.expand_dims(const, 1) 


def echo_capacity(inputs, init = -5, batch = 100):  
# needed to pass the capacity parameter to a loss function

  if isinstance(inputs, list):
    z_mean = inputs[0] # only one stat argument to echo sample (mean, no variance)
  else:
    z_mean = inputs
  #if isinstance(inputs, list):
  #  z_mean = inputs[0] # only one stat argument to echo sample (mean, no variance)
  #else:
  #  z_mean = inputs
  with tf.variable_scope('encoder_noise', reuse=tf.AUTO_REUSE):
    #latent_shape = z_mean.get_shape().as_list()[1:] if not hasattr(z_mean, '_keras_shape') else z_mean._keras_shape[1:]
    #init = tf.constant(init, shape=latent_shape, dtype=tf.float32)  # Init with very small noise
    cap_param = K.variable(tf.get_variable("capacity_parameter"))#, initializer=init)
    #phi = tf.get_variable('phi', initializer=tf.constant(np.pi, shape=latent_shape, dtype=tf.float32))
    #c = tf.sigmoid(cap_param, name="e_cap")
    return cap_param

def echo_sample(inputs, init = -5., d_max = 50, batch = 100, multiplicative = False, 
      noise = 'additive', trainable = True, periodic = False):

  if isinstance(inputs, list):
    z_mean = inputs[0] # only one stat argument to echo sample (mean, no variance)
  else:
    z_mean = inputs
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

  #batch = z_mean.get_shape().as_list()[0] if not hasattr(z_mean, '_keras_shape') else K.int_shape(z_mean)[0] #K.cast(K.shape(z_mean)[0], K.floatx()) 
  with tf.variable_scope('encoder_noise', reuse=tf.AUTO_REUSE):
    latent_shape = z_mean.get_shape().as_list()[1:] if not hasattr(z_mean, '_keras_shape') else z_mean._keras_shape[1:]
    init = tf.constant(init, shape=latent_shape, dtype=tf.float32)  # Init with very small noise
    cap_param = K.variable(tf.get_variable("capacity_parameter", initializer=init, trainable = trainable))
    tf.Session().run(cap_param.initializer)
    phi = tf.get_variable('phi', initializer=tf.constant(np.pi, shape=latent_shape, dtype=tf.float32))
    c = tf.sigmoid(cap_param, name="e_cap")
    
    inds = permute_neighbor_indices(batch, d_max)

    inds = tf.constant(inds, dtype=tf.int32)
    if multiplicative or noise == 'multiplicative':
        normal_encoder = z_mean #tf.log(z_mean + 1e-5) # noise calc done in log space
    else:
        normal_encoder = z_mean

    if periodic:
      c_z_stack = tf.stack([tf.cos(k * phi) * tf.pow(c, k) * normal_encoder for k in range(d_max)])
    else:
      c_z_stack = tf.stack([tf.pow(c, k) * normal_encoder for k in range(d_max)])  # no phase
    
    #noise = tf.gather(c_z_stack, inds, axis = 0)
    noise = tf.gather_nd(c_z_stack, inds)
    noise = tf.reduce_sum(noise, axis=1)  # Sums over d_max terms in sum
    noise -= tf.reduce_mean(noise, axis=0)  # Add constant so batch mean is zero

    if multiplicative:
        noisy_encoder = tf.exp(z_mean + c*noise)# #z_mean * tf.exp(c * noise)
    else:
        noisy_encoder = z_mean + c * noise

    return noisy_encoder  

def my_predict(model, data, layer_name, multiple = True):
        func = K.function([model.layers[0].get_input_at(0)],
                        [model.get_layer(layer_name).get_output_at(0)])
        return func([data])[0]

def positive_log_prob(dist, x):
  #, event_ndims=0
  return (dist.bijector.inverse_log_det_jacobian(x, 1) +
          dist.distribution.log_prob(dist.bijector.inverse(x)))

def negative_log_prob(dist, x):
  #, event_ndims=0
  return -(dist.bijector.inverse_log_det_jacobian(x, 1) +
          dist.distribution.log_prob(dist.bijector.inverse(x)))

def tf_masked_flow(inputs, steps = None, layers = None, mean_only = True, activation = None, name = 'maf'):
  if isinstance(inputs, list):
    z_mean = inputs[0]
  else:
    z_mean = inputs
  dim = K.int_shape(z_mean)[-1]

  #This doesn't really make sense
  if steps is not None and layers is None: 
    layers = [dim]*steps
  steps = steps if steps is not None else 1

  # if inverse:
  #   iaf = tfd.TransformedDistribution(
  #       distribution=tfd.Normal(loc=z_mean, scale=z_std),
  #       bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
  #           shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
  #               hidden_layers= layers, shift_only= mean_only))),  name = name,
  #       event_shape=[dim])
  # else:
  #for step in steps:
  layers = tuple(layers)
  print("Input ", z_mean)
  maf_chain = list(itertools.chain.from_iterable([
          tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
            hidden_layers = layers, shift_only = mean_only), name = name+str(i))
            #, **{"kernel_initializer": tf._initializer()}))
          ,tfb.Permute(list(reversed(range(dim))))]  #np.random.permutation(dim)
          for i in range(steps)))
  print("maf chain ", maf_chain)
  #tfd.Normal(loc=0.0, scale=1.0),
  maf = tfd.TransformedDistribution(
    distribution= tfd.MultivariateNormalDiag(
      loc=tf.zeros([dim]), allow_nan_stats = False),
    bijector=tfb.Chain(maf_chain[:-1])
    #, validate_args = True)
    )
  #print()
  #print("TRANSFORMED PROB SHAPE ", negative_log_prob(maf, z_mean))
  #print()
  # maf = tfd.TransformedDistribution(
  #   distribution=tfd.Normal(loc=0.0, scale=1.0),
  #   bijector=tfb.MaskedAutoregressiveFlow(
  #       shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
  #           hidden_layers=layers, shift_only=mean_only), activation = activation), name = name+step)
  print("maf name ", maf.name)
  print("z mean name ", z_mean.name)
  return K.expand_dims(negative_log_prob(maf, z_mean), 1)
  #transformed_log_prob(maf, z_mean)
  #return maf.bijector.log_prob(z_mean)



# def tf_inverse_flow(inputs, steps = None, layers = None, mean_only = True, name = None):
#   z_mean, z_logvar = inputs

#   dim = K.int_shape(z_mean)[-1]
#   if steps is not None and layers is None:
#     layers = [dim]*steps

#   if name is None:
#     name = 'iaf'

#   # uses relu as default
#   iaf = tfd.TransformedDistribution(
#       distribution= tfd.MultivariateNormalDiag(loc = z_mean,  
#         scale_diag =tf.exp(.5*z_logvar)),
#       bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
#           shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
#               hidden_layers= layers, shift_only= mean_only))),  name = name,
#       event_shape=[dim])

#   #iaf.sample()
#   #iaf.log_prob(sample)
#   return iaf




class Echo(Layer):
  def __init__(self, init = -5., batch = 200, d_max = 50, trainable = False, noise = 'additive', multiplicative = False, periodic = False, **kwargs):
      self.init = init
      self.trainable = trainable
      self.noise = noise if not (multiplicative and noise == "additive") else 'multiplicative'

      self.d_max = d_max
      self.periodic = periodic
      self.batch = batch
      #self.name = name
      super(Echo, self).__init__(**kwargs)
      #super([Layer], self).__init__()

  def build(self, input_shape):
      #self.batch = input_shape[0]
      if isinstance(input_shape, list):
        self.dim = input_shape[0][-1]
      else:
        self.dim = input_shape[-1]

      #print("BATCH SIZE ECHO LAYER ", self.batch)
      if self.trainable:
        self.cap_param = self.add_weight(name='capacity', 
                                      shape = (self.dim,),
                                      initializer= Constant(value = self.init),
                                      trainable= True)
      else:
        self.cap_param = self.add_weight(name='capacity', 
                              shape = (self.dim,),
                              initializer= Constant(value = self.init),
                              trainable= False)
      super(Echo, self).build(input_shape)
      #super(Beta, self).build(input_shape) 

  def call(self, z_mean):
    #print("Z mean type ", z_mean)
    if isinstance(z_mean, list):
      z_mean = z_mean[0]
    #z_mean = z_mean[0]
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

    c = K.sigmoid(self.cap_param)#, name = 'capacity0-1')
    inds = permute_neighbor_indices(self.batch, self.d_max)


    if self.periodic:
      c_z_stack = tf.stack([tf.cos(k * phi) * tf.pow(c, k) * z_mean for k in range(self.d_max)])
    else:
      c_z_stack = tf.stack([tf.pow(c, k) * z_mean for k in range(self.d_max)])  # no phase
    

    noise = tf.gather_nd(c_z_stack, inds)
    noise = tf.reduce_sum(noise, axis=1)  # Sums over d_max terms in sum
    noise -= tf.reduce_mean(noise, axis=0)

    if self.noise == 'multiplicative':
        noisy_encoder = tf.exp(z_mean + c*noise)# #z_mean * tf.exp(c * noise)
    else:
        noisy_encoder = z_mean + c * noise

    self.prev_noise = noise
    noisy_encoder.set_shape(K.int_shape(z_mean))
    #self.cap_param.set_shape()
    return noisy_encoder
    #return [noisy_encoder, self.cap_param] 
    #K.repeat_elements(K.expand_dims(self.betas,1), repeat, -1)

  def compute_output_shape(self, input_shape):
      return input_shape
      #return [input_shape, K.int_shape(self.cap_param)]
      #(input_shape[0], self.dim)

  def get_capacity(self, x):
      return self.cap_param
  
  def get_noise(self,x):
      return self.noise

def make_ar_flow(x, transform_dims = [640, 640, 640, 640], activation = 'relu', flow_type = 'planar', mask = 'made', prefix = ''):
  for i in range(len(transform_dims)):
    x = Dense(transform_dims[i], activation = activation, name = 'flow_'+prefix+i)(x)
    x = Lambda(flow_transform, arguments = {"flow_type": flow_type})(x)
    if mask == 'made':
      pass

class IAF(Layer):
  def __init__(self, steps = None, layers = None, activation = 'relu', mean_only = True, name = 'iaf', **kwargs):
    
    self.layers = layers
    self.steps = steps if steps is not None else 1
    self.mean_only = mean_only
    self.name = name
    try:
      mod = importlib.import_module('keras.activations')
      self.activation = getattr(mod, self.activation)
    except:
      try:
        mod = importlib.import_module(self.activation)
        self.activation = getattr(mod, self.activation.split(".")[-1])
      except:
        self.activation = activation
    
    super(IAF, self).__init__(**kwargs)

  def build(self, input_shape):
    # input shape is list of [z_mean, z_std]
    if isinstance(input_shape, list):
      self.dim = input_shape[0][-1]
    else:
      self.dim = input_shape[-1]

    if self.layers is None:
      self.layers = [self.dim, self.dim, self.dim]

    iaf_chain = list(itertools.chain.from_iterable([
      tfb.Invert(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=self.layers, shift_only=self.mean_only, name = self.name+str(i)))),
              #**{"kernel_initializer": tf.ones_initializer()}))),
      tfb.Permute(list(reversed(range(self.dim))))] #)
            for i in range(self.steps)))

    self.bijector = tfb.Chain(iaf_chain[:-1]) 


      # tfb.MaskedAutoregressiveFlow(
      #   shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
      #   hidden_layers = layers, shift_only = mean_only, activation = activation, name = name+str(i))),
      # tfb.Permute(list(reversed(range(dim))))] 
      # for i in range(steps)))

    # HOW TO INCORPORATE (loc=z_mean, scale=z_std)?
    # self.iaf = tfd.TransformedDistribution(
    #     distribution= tfd.MultivariateNormalDiag(loc = z_mean,  
    #     scale_diag =tf.exp(.5*z_logvar)),
    #     bijector = )

      # tfb.Invert(tfb.MaskedAutoregressiveFlow(
      #     shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
      #         hidden_layers=self.layers, shift_only=self.mean_only))),  name = self.name,
      # event_shape=[self.dim])

    self.built = True
    
  def call(self, inputs):
    z_mean, z_logvar = inputs
    print("input shape ", z_mean.get_shape().as_list())
    self.iaf =  tfd.TransformedDistribution(
        distribution = tfd.MultivariateNormalDiag(loc = z_mean,  
        scale_diag = tf.exp(.5*z_logvar)),
        bijector = self.bijector)
    
    last_samples = self.iaf.sample()
    print('sample size ', last_samples.get_shape().as_list() )
    self.density = self.iaf.log_prob(last_samples)
    print('density size ', self.density.get_shape().as_list() )
    try:
      #self.density = self.iaf.log_prob(last_samples)
      self.density = K.squeeze(K.squeeze(self.iaf.log_prob(last_samples), 1),1)
    except:
      self.density = self.iaf.log_prob(last_samples)
    print('density size ', self.density.get_shape().as_list() )
    return (last_samples)

  def get_density(self, x):
    return K.expand_dims(self.density, 1) 

  def compute_output_shape(self, input_shape):
    # CHECK THIS, PROBABLY NOT GENERAL
    return input_shape[0]

  def get_log_det_jac(self, x):
    return self.iaf.inverse_log_det_jacobian(x, 0)


# class Inverse_AR_Flow(Layer):
#   def __init__(self, layers = 0, transform_dims = [], activation = 'relu', mask = 'made', **kwargs):
#     self.layers = layers
#     self.transform_dims = transform_dims
#     if layers == 0 and isempty(transform_dims):
#       raise ValueError("Please enter # of dims for inverse autoregressive flow")
#     try:
#       mod = importlib.import_module('keras.activations')
#       self.activation = getattr(mod, self.activation)
#     except:
#       self.activation = activation

#     self.mask = mask
#     self.w = []
#     self.u = []
#     self.b = []
#     super(Inverse_AR_Flow, self).__init__(**kwargs)

#     def build(self, input_shape):
#       # only variances figure into loss
#       self.dim = input_shape[-1]
#       if not isinstance(self.layers[0], list):
#         self.layers = [self.layers.append(self.dim) for i in range(self.steps)]

#       self.made_layers = defaultdict(lambda: defaultdict(list))
#       for i in range(len(self.layers)):

#         #if i == len(self.layers)-1:
#         #  self.made_layers[i]['act'].append(
#         #    Lambda(vae_sample))
#         print("MADE network layer SIZE: ", self.layers[i])
#         mean_made = MADE(out_units = self.dim, hidden_dims = self.layers[i], 
#                 random_input_order = True, activation = 'relu' if self.activation is None else self.activation, out_activation = 'linear', 
#                 name = 'made_mean_'+str(i))
#         self.made_layers[i]['stat'].append(mean_made)
#         made_std = MADE(out_units = self.dim, hidden_dims = self.layers[i], 
#           random_input_order = True, activation = 'relu' if self.activation is None else self.activation, out_activation = 'linear',
#           name = 'made_std_'+str(i))
#             #stat_list.append([made_mean, made_std])
  
#         self.made_layers[i]['stat'].append(made_std)
#         # NEEDS previous activation AS INPUT 
#         maf_act = Lambda(inverse_vae_sample, name = 'made_noise'+str(i))
#         self.made_layers[i]['act'] = maf_act
#         #self.made_layers[i]['act'].append(maf_act)
#       self.built = True
    
#     def call(self, x):
#       self.made_tensors = defaultdict(lambda: defaultdict(list))
#       # e.g. x = target z 
#       for i in range(self.steps):
#         m, s = self.made_layers[i]['stat']
#         self.made_tensors[i]['stat'] = [m(x), s(x)]
#         self.made_tensors[i]['act'] = self.made_layers[i]['act']([x, *self.made_tensors[i]['stat']])
#         x = self.made_tensors[i]['act'] 
#       return x

#     def compute_output_shape(self, input_shape):
#       # CHECK THIS, PROBABLY NOT GENERAL
#       return input_shape

#     def get_log_det_jac(self, x):
#       return K.sum(tf.add_n([-.5*self.made_tensors[i]['stat'][-1] for i in range(self.steps)]), axis = -1, keepdims= True)
#       #K.sum(-.5*log_var, axis = -1)
#       #K.sum([self.made_tensors[i]['stat'][-1] for i in range(self.steps)])
#     def get_alpha_i(self, x):
#       return self.get_log_det_jac(x)


class MADE_network(Layer):
    def __init__(self, steps, layers, mean_only = True, **kwargs):# inpuy tensor placeholder for 
      self.layers = layers
      try: 
        self.activation = activation
      except:
        self.activation = None

      self.steps = steps
      self.mean_only = mean_only
      super(MADE_network, self).__init__(**kwargs)

    def build(self, input_shape):
      # only variances figure into loss
      self.dim = input_shape[-1]
      if not isinstance(self.layers[0], list):
        self.layers = [self.layers.append(self.dim) for i in range(self.steps)]

      self.made_layers = defaultdict(lambda: defaultdict(list))
      for i in range(len(self.layers)):

        #if i == len(self.layers)-1:
        #  self.made_layers[i]['act'].append(
        #    Lambda(vae_sample))
        print("MADE network layer SIZE: ", self.layers[i])
        mean_made = MADE(out_units = self.dim, hidden_dims = self.layers[i], 
                random_input_order = True, activation = 'relu' if self.activation is None else self.activation, out_activation = 'linear', 
                name = 'made_mean_'+str(i))
        self.made_layers[i]['stat'].append(mean_made)
        #try:
        #    # loss mean only?
        #    a = self.mean_only
        #except:
        #    mean_only = self.loss_kwargs.get('mean_only', False)
        if not self.mean_only:
            made_std = MADE(out_units = self.dim, hidden_dims = self.layers[i], 
              random_input_order = True, activation = 'relu' if self.activation is None else self.activation, out_activation = 'linear',
              name = 'made_std_'+str(i))
            #stat_list.append([made_mean, made_std])
        else:
            made_std = K.constant(1.0)
            #stat_list.append([made_mean])
        self.made_layers[i]['stat'].append(made_std)
        # NEEDS previous activation AS INPUT 
        maf_act = Lambda(inverse_vae_sample, name = 'made_noise'+str(i))
        self.made_layers[i]['act'] = maf_act
        #self.made_layers[i]['act'].append(maf_act)
      self.built = True
    
    def call(self, x):
      self.made_tensors = defaultdict(lambda: defaultdict(list))
      # e.g. x = target z 
      for i in range(self.steps):
        m, s = self.made_layers[i]['stat']
        print(m,s)
        self.made_tensors[i]['stat'] = [m(x), s(x)]
        self.made_tensors[i]['act'] = self.made_layers[i]['act']([x, *self.made_tensors[i]['stat']])
        x = self.made_tensors[i]['act'] 
      return x

    def compute_output_shape(self, input_shape):
      # CHECK THIS, PROBABLY NOT GENERAL
      return input_shape

    def get_log_det_jac(self, x):
      return K.sum(tf.add_n([-.5*self.made_tensors[i]['stat'][-1] for i in range(self.steps)]), axis = -1, keepdims= True)
      #K.sum(-.5*log_var, axis = -1)
      #K.sum([self.made_tensors[i]['stat'][-1] for i in range(self.steps)])
    def get_alpha_i(self, x):
      return self.get_log_det_jac(x)

class MADE(Layer):
    """ Just copied code from keras Dense layer and added masking """
    """ taken from https://github.com/bjlkeng/sandbox/tree/master/notebooks/masked_autoencoders """
    def __init__(self, out_units = None,
                 hidden_dims = None,
                 hidden_layers=1,
                 dropout_rate=0.0,
                 random_input_order=False,
                 fc=False,
                 activation='relu',
                 out_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MADE, self).__init__(**kwargs)
        
        self.input_sel = None
        self.random_input_order = random_input_order
        self.rate = min(1., max(0., dropout_rate))
        self.kernel_sels = []
        #self.units = units
        self.out_units = out_units
        self.hidden_layers = hidden_layers
        if hidden_dims is None:
          try:
            self.hidden_dims = [self.out_units]*self.hidden_layers
          except:
            pass
        else:
          self.hidden_layers = len(hidden_dims)
          self.hidden_dims = hidden_dims

        self.fc = fc
        #gaussian_inputs=False,
        #self.gaussian_inputs = gaussian_inputs

        self.final_activation=activations.get(activation)
        self.activation = activations.get(activation)
        self.out_activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
    def dropout_wrapper(self, inputs, training):
        if 0. < self.rate < 1.:
            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape=None, seed=None)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        
        return inputs
        
    def build_layer_weights(self, input_dim, units, use_bias=True):
        kernel = self.add_weight(shape=(input_dim, units),
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
     
        if use_bias:
            bias = self.add_weight(shape=(units,),
                                   initializer=self.bias_initializer,
                                   name='bias',
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        else:
            bias = None
        
        return kernel, bias
    
    def build_mask(self, shape, prev_sel, is_output):
        if is_output:
            input_sel = self.input_sel
        else:
            # Disallow D because it would violate auto-regressive property
            # Disallow 0 because it would just createa a constant node
            # Disallow unconnected units by sampling min from previous layer
            input_sel = [randint(np.min(prev_sel), shape[-1] - 1) for i in range(shape[-1])]
            
        def vals():
            for x in range(shape[-2]):
                for y in range(shape[-1]):
                    if is_output:
                        yield 1 if prev_sel[x] < input_sel[y] else 0
                    else:
                        yield 1 if prev_sel[x] <= input_sel[y] else 0
        
        return K.constant(list(vals()), dtype='float32', shape=shape), input_sel
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
           
        self.kernels, self.biases = [], []
        self.kernel_masks, self.kernel_sels = [], []
        
       
        # RANDOMIZATION OF INPUT ORDERING
        self.input_sel = np.arange(input_shape[-1])
        if self.random_input_order:
            np.random.shuffle(self.input_sel)

        prev_sel = self.input_sel
        
        for x in range(self.hidden_layers):
            #if x == 0:
             #self.units)
            shape = (input_shape[-1], self.hidden_dims[x]) if x == 0 else (self.hidden_dims[x-1], self.hidden_dims[x])
            # Hidden layer
            kernel, bias = self.build_layer_weights(*shape)
            self.kernels.append(kernel)
            self.biases.append(bias)
            
            # Hidden layer mask
            kernel_mask, kernel_sel = self.build_mask(shape, prev_sel, is_output=self.fc)
            self.kernel_masks.append(kernel_mask)
            self.kernel_sels.append(kernel_sel)
        
            prev_sel = kernel_sel
            #shape = (self.units, self.units)
            
        # Direct connection between input/output
        direct_shape = (input_shape[-1], self.out_units)
        self.direct_kernel, _ = self.build_layer_weights(*direct_shape, use_bias=False)
        self.direct_kernel_mask, self.direct_sel = self.build_mask(direct_shape, self.input_sel, is_output=True)
        
        #self.units
        # Output layer
        out_shape = (self.hidden_dims[-1], self.out_units)
        self.out_kernel, self.out_bias = self.build_layer_weights(*out_shape)
        self.out_kernel_mask, self.out_sel = self.build_mask(out_shape, prev_sel, is_output=True)
        
        self.built = True

    def call(self, inputs, training=None):
        #if self.gaussian_inputs:
        #  output = K.random_normal(shape = (inputs._keras_shape[-1],), mean = 0.0, stddev= 1.0)
        #  inputs = tf.identity(outputs)
        #else:
        # Hidden layer + mask
        output = inputs

        for i in range(self.hidden_layers):
            weight = self.kernels[i] * self.kernel_masks[i]
            output = K.dot(output, weight)
            output = K.bias_add(output, self.biases[i])
            output = self.activation(output)
            output = self.dropout_wrapper(output, training)
       
        # Direct connection
        direct = K.dot(inputs, self.direct_kernel * self.direct_kernel_mask)
        direct = self.dropout_wrapper(direct, training)
        
        # out_act(bias + (V dot M_v)h(x) + (A dot M_v)x) 
        output = K.dot(output, self.out_kernel * self.out_kernel_mask)
        output = output + direct
        output = K.bias_add(output, self.out_bias)
        output = self.out_activation(output)
        
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_units)

class AR_Flow(Layer):
  def __init__(self, layers = 0, dims = [], activation = 'relu', mask = 'made', **kwargs):
    self.layers = layers
    self.transform_dims = dims
    if layers == 0 and isempty(dims):
      raise ValueError("Please enter # of dims for autoregressive flow")
    
    try:
      mod = importlib.import_module('keras.activations')
      self.activation = getattr(mod, self.activation)
    except:
      self.activation = activation


    self.mask = mask
    self.w = []
    self.u = []
    self.b = []
    super(AR_Flow, self).__init__(**kwargs)
    
  def build(self, input_shape):
  
    self.dim = input_shape[-1]
    dims = layers if layers != 0 else len(self.transform_dims)


    for i in range(len(dims)):
      if layers == 0: 
        self.dim = transform_dims[i]
      
      self.w.append(self.add_weight(name='w_'+i, 
                            shape = (self.dim,),
                            initializer= 'glorot_uniform',
                            trainable= True))
      #x = Dense(transform_dims[i], activation = activation, name = 'flow_'+prefix+i)(x)
      
      self.u.append(self.add_weight(name='u_'+i, 
                            shape = (self.dim,),
                            initializer= 'glorot_uniform',
                            trainable= True))

      self.b.append(self.add_weight(name='b_'+i, 
                            shape = (1,),
                            initializer= 'zero',
                            trainable= True))

      super(AR_Flow, self).build(input_shape)
      #x = Lambda(flow_transform, arguments = {"flow_type": flow_type})(x)
      
  def call(self, x):
    logdetjac = 0
    x = K.random_normal(shape=z_mean._keras_shape[-1],
                                  mean=0.,
                                  stddev=1.0)
    log_q0 = 2*np.pi**(-.5)*K.exp(.5*x**2)

    for i in range(len(self.w)):
      h = self.activation(x*self.w[i] + self.b[i])
      psi = h*self.w[i]
      logdetjac += K.log(1+K.transpose(self.u[i])*psi)
      x = x + u*h
    self.density = log_q0 - logdetjac
    return x #log_q0 - logdetjac

  def compute_output_shape(self, input_shape):
      return input_shape  

  def get_density(self, x):
      return self.density



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



