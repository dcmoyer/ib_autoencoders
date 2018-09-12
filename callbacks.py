from keras.callbacks import Callback
import tensorflow as tf
from keras.backend import get_session


class BetaCallback(Callback):
    def __init__(self, functions, layers):#betas, sched, screening = False, recon = False): # mod, minbeta=0):
        self.layers = layers
        self.anneal_functions = functions
    
    def on_epoch_begin(self, epoch, logs={}):
        for l in range(len(self.layers)): 
            tf.assign(self.layers[l],self.anneal_functions[l](epoch)).eval(session=get_session())
            #print("anneal value ", self.layers[l])
            