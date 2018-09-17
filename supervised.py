import keras
import keras.models
import keras.backend as K
import tensorflow as tf
import dataset 
from layers import Echo, vae_sample
from keras.layers import Input, Dense, merge, Lambda, Flatten #Concatenate, 
from keras.layers import Activation, BatchNormalization, Lambda, Reshape

method = 'echo'
d = dataset.MNIST(binary = True)
a = 'softplus'
x = Input(shape=(784,))
d1 = Dense(200, activation = a)(x)
d2 = Dense(200, activation = a)(d1)
if method == 'vae':
	z_mean = Dense(50, activation = 'linear')(d2)
	z_logvar = Dense(50, activation = 'linear')(d2)
	z_act = Lambda(vae_sample)([z_mean, z_logvar])
else:
	z_mean = Dense(50, activation = 'linear')(d2)
	z_act = Echo(trainable = True)(z_mean)

d3 = Dense(200, activation = a)(z_act)
d4 = Dense(200, activation = a)(d3)
output = Dense(10, activation = 'softmax')(d4)
m = keras.models.Model(inputs = x, outputs = output)
m.compile(optimizer = 'adam', loss='categorical_crossentropy')
y = keras.utils.to_categorical(d.y_train[:50000], 10)
yt = keras.utils.to_categorical(d.y_test, 10)
m.fit(d.x_train, y, batch_size=200, epochs=200, verbose=True)
print(m.evaluate(d.x_test, yt, batch_size=200, verbose=True))