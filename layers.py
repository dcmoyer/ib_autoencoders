import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda, merge, Dense
from keras import activations
from keras.initializers import Constant
import numpy as np

def vae_sample(args):
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
  z_mean, z_noise = args
  std = 1.0
  z_score = K.random_normal(shape=(z_mean._keras_shape[1],),
                                  mean=0.,
                                  stddev=std)
    
  return K.exp(K.log(z_mean) + K.exp(z_noise / 2) * z_score)
  #return K.exp(K.log(z_mean) + z_noise * epsilon)

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

