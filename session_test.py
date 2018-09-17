import session
import dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test_config.json')
parser.add_argument('--name', type=str, default='latest_unnamed_session')
parser.add_argument('--time', type=str)
parser.add_argument('--verbose', type=bool, default = 1)
parser.add_argument('--per_label', type=int)
parser.add_argument('--dataset', type=str, default='binary_mnist')
#parser.add_argument('--echo_init', type=float)                                                                                                                               
#parser.add_argument('--noise', type=str)                                                                                                                                     
#parser.add_argument('--constraint', type=float)                                                                                                                              
args, _ = parser.parse_known_args()

#params = {"constraints.0.value":[15, 10, 5]}                                                                                                                                 


#betas = [0.05, 0.01,0.001,0.4,0.3,0.2,20,10,9,8,7,6,5,4,3,2,1.5,1.4,1.3,1.2,1.15,1.1,1.05,1,0.95,0.9,0.8,0.7,0.6,0.5,0.1,0]                                                  
#betas = [0,.5,1,2]                                                                                                                                                           
betas = [0.0, .0005, .0001, .001, .005, .01, .1, .25, .5, .7, .8, .9, 1.0, 1.1, 1.3, 1.5, 2, 4, 8]
#betas = [.1, .25, .5, .7, .8, .9, 1.0, 1.1, 1.3, 1.5, 2]
betas = [b*-1 for b in betas]
beta_params = {"losses.0.weight": betas, "losses.1.weight":betas}
vary_together = True

mis = [50.0,65.0,80.0,95.0,110,0.58,1.32,2.97,3.57,4.9,6.09,8.14,10.02,11.75,13.18,15.55,17.14,19.56,22,25,28.19,35]
mi_params = {"constraints.0.value": mis}

params_used = beta_params # mi_params                                                                                                                                         

#params = {"activation.encoder": ["softplus", "sigmoid"]}                                                                                                                     
if args.dataset == 'fmnist':
	d = dataset.fMNIST()
elif args.dataset == 'binary_mnist':
	d = dataset.MNIST(binary= True)
elif args.dataset == 'mnist':
	d = dataset.MNIST()

if args.per_label is not None:
	d.shrink_supervised(args.per_label)


# name is important!  = filesave location                                                                                                                                     
if args.time is not None:
    a = session.Session(name=args.name, config=args.config, dataset = d, parameters= params_used, time = args.time, verbose = args.verbose, per_label = args.per_label, vary_together= vary_together)
else:
    a = session.Session(name=args.name, config=args.config, dataset = d, parameters= params_used, verbose = args.verbose, per_label = args.per_label, vary_together = vary_together)
#for i in a.configs:
#	print(i)
#	print()
#print(a.configs)
