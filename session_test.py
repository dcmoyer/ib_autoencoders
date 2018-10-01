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
parser.add_argument('--num_losses', type=int, default= 1)                                                                                                                               
parser.add_argument('--params', type=str, default= 'betas')     
parser.add_argument('--dmax', type=int, nargs='+') #parser.add_argument('--noise', type=str)                                                                                                                                     
#parser.add_argument('--constraint', type=float)                                                                                                                              
args, _ = parser.parse_known_args()

#params = {"constraints.0.value":[15, 10, 5]}                                                                                                                                 


#betas = [0.05, 0.01,0.001,0.4,0.3,0.2,20,10,9,8,7,6,5,4,3,2,1.5,1.4,1.3,1.2,1.15,1.1,1.05,1,0.95,0.9,0.8,0.7,0.6,0.5,0.1,0]                                                  
#betas = [0,.5,1,2]              
if args.params in ['few_betas', 'small_beta']:
        betas = [1.0, 2, 0.1, 0.5]
else:
        #betas = [0, 0.125, 0.0005, 8, 4, 0.25, 2, 1.1, 1.75, 0.025, 0.175, 0.05, 0.5, 1.0, 0.15, 0.4, 0.225, 0.01, 0.1, 0.9, 0.0075, 0.075, 0.7, 0.3, 0.2, 1.5, 1.3]
        betas = [0, 0.125, 0.25, 2, 1, 1.5, .025, .175, .15, 0.05, .5, .3, .225, .2, .01, .75, 1.25, 0.075, 4]


mis = [50.0,65.0,80.0,95.0,110,0.58,1.32,2.97,3.57,4.9,6.09,8.14,10.02,11.75,13.18,15.55,17.14,19.56,22,25,28.19,35]

if args.per_label is not None:
        args.per_label = int(args.per_label)

if args.num_losses > 1:
	beta_params = {"losses.0.weight": betas, "losses.1.weight":betas}
	mi_params = {"constraints.0.value": mis, "constraints.1.value": mis}
else:
	beta_params = {"losses.0.weight": betas}
	mi_params = {"constraints.0.value": mis}
vary_together = True


if args.params in ['beta', 'betas', 'b']:
	params_used = beta_params # mi_params                                                                                                                                         
elif args.params in ['mi', 'mis', 'm', 'constraint', 'constraints']:
	params_used = mi_params
elif args.params in ['few_betas', 'small_beta']:
        params_used = beta_params

if args.dmax is not None:
        params_used["layers.5.layer_kwargs.d_max"] = list(args.dmax)
        vary_together = False

#params = {"activation.encoder": ["softplus", "sigmoid"]}                                                                                                                     
if args.dataset == 'fmnist':
	d = dataset.fMNIST()
elif args.dataset == 'binary_mnist':
	d = dataset.MNIST(binary= True)
elif args.dataset == 'mnist':
	d = dataset.MNIST()
elif args.dataset == 'omniglot':
	d = dataset.Omniglot()
elif args.dataset == 'dsprites':
	d = dataset.DSprites()

if args.per_label is not None:
	d.shrink_supervised(int(args.per_label))


# name is important!  = filesave location                                                                                                                                     
if args.time is not None:
    a = session.Session(name=args.name, config=args.config, dataset = d, parameters= params_used, time = args.time, verbose = args.verbose, per_label = args.per_label, vary_together= vary_together)
else:
    a = session.Session(name=args.name, config=args.config, dataset = d, parameters= params_used, verbose = args.verbose, per_label = args.per_label, vary_together = vary_together)
#for i in a.configs:
#	print(i)
#	print()
#print(a.configs)



