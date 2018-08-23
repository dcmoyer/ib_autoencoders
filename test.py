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
parser.add_argument('--echo_init', type=float)
parser.add_argument('--noise', type=str)
args, _ = parser.parse_known_args()

print("ECHO INIT ", args.echo_init)
print("NOISE ", args.noise)

d = dataset.MNIST(binary= True)
#i = Input(shape = (d.x_train.shape[-1],))
#l = Dense(10)(i)
#print(l, callable(l))
#l = Dense(10)
#l = l(i)
#print(l, callable(l))

init_str = str(args.echo_init if args.echo_init is not None else "")
noise_str = str(args.noise if args.noise is not None else "")
fn = str("echo"+init_str+noise_str)

m = model.NoiseModel(d, config = 'test_config.json', filename = fn)


if args.echo_init is not None: 
	m.layers[0]['layer_kwargs']['init'] = args.echo_init
if args.noise is not None:
	if args.noise == 'multiplicative':
		m.layers[0]['layer_kwargs']['multiplicative'] = True
	if args.noise == 'additive':
		m.layers[0]['layer_kwargs']['multiplicative'] = False

m.fit(d.x_train)
