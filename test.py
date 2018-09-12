import importlib
import model
import dataset
import layers 
import loss_args
import losses
from keras.layers import Dense, Input
import argparse
import json
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test_config.json')
parser.add_argument('--echo_init', type=float)
parser.add_argument('--noise', type=str)
parser.add_argument('--constraint', type=float)
parser.add_argument('--filename', type=str)
parser.add_argument('--beta', type=float)
args, _ = parser.parse_known_args()

print("ECHO INIT ", args.echo_init)
print("NOISE ", args.noise)
print("Constraint ", args.constraint)

if ".json" in args.config:
	config = args.config
else:
	#config = ast.literal_eval(args.config)
	print(args.config)
	config = json.loads(args.config.replace("'", '"')) 


print("replaced config")
print(config)

d = dataset.fMNIST()
#d = dataset.MNIST(binary= True)
#i = Input(shape = (d.x_train.shape[-1],))
#l = Dense(10)(i)
#print(l, callable(l))
#l = Dense(10)
#l = l(i)
#print(l, callable(l))


init_str = str(args.echo_init if args.echo_init is not None else "")
noise_str = str(args.noise if args.noise is not None else "")
constr = str(args.constraint if args.constraint is not None	else "")
beta_str = str(args.beta if args.beta is not None else "")
if args.filename is None:
	fn = str("echo_"+init_str+noise_str+constr+beta_str)
else:
	fn = args.filename


m = model.NoiseModel(d, config = config, filename = fn)


if args.echo_init is not None: 
	m.layers[0]['layer_kwargs']['init'] = args.echo_init
if args.noise is not None:
	if args.noise == 'multiplicative':
		m.layers[0]['layer_kwargs']['multiplicative'] = True
	if args.noise == 'additive':
		m.layers[0]['layer_kwargs']['multiplicative'] = False
if args.constraint is not None:
	m.constraints[0]['value'] = args.constraint
if args.beta is not None:
	m.losses[0]['weight'] = args.beta
	
m.fit(d.x_train)
