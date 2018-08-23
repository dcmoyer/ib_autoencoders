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
for mi_init in [1,2]: 
	m = model.NoiseModel(d, config = 'test_config.json', filename = str('mi_'+str(round(mi_init,2))))
	m.layers[0]['layer_kwargs']['init'] = mi_init
	m.fit(d.x_train)
	tf.reset_default_graph()