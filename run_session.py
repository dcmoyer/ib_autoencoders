import session
import dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test_config.json')
parser.add_argument('--echo_init', type=float)
parser.add_argument('--noise', type=str)
parser.add_argument('--constraint', type=float)
args, _ = parser.parse_known_args()

params = {"constraints.0.value":[15, 10, 5]}
#params = {"activation.encoder": ["softplus", "sigmoid"]}
d = dataset.MNIST(binary= True)
a = session.Session(name='test', config=args.config, dataset = d, parameters=params)

#for i in a.configs:
#	print(i)
#	print()
#print(a.configs)