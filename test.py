import importlib
import model
import dataset
import layers 
import loss_args
import losses
from keras.layers import Dense, Input
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test_config.json')

d = dataset.MNIST(binary= True)
#i = Input(shape = (d.x_train.shape[-1],))
#l = Dense(10)(i)
#print(l, callable(l))
#l = Dense(10)
#l = l(i)
#print(l, callable(l)) 
m = model.NoiseModel(d, config = 'test_config.json')
m.fit(d.x_train)