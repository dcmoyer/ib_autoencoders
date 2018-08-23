import numpy as np
import keras.backend as K
import importlib
import tensorflow as tf
import losses as l
import mixture_losses as m
from keras.layers import Lambda
#from model import RECON_LOSSES

RECON_LOSSES = ['bce', 'mse', 'binary_crossentropy', 'mean_square_error', 'mean_squared_error', 'iwae']
LOSS_WEIGHTS = 'loss_weights' # path to loss weights module

class Loss(object):

    def __init__(self, beta = None, **kwargs):
        ''' 
        expected kwargs:
        {
        "layer": -1, /* 0, 1, 2 indexing for encoder layers.  (d0, d1, d2 for decoder? not currently supported)  */
        "encoder": true,
        "type": "add",          
        "k": 1,
        "add_loss": true,
        "output": -1 /* output should be here or in noise specification? */
        }
        '''
        args = {
            'type': 'vae',
            'layer': -1,
            'encoder': True,
            'weight': 1, # float or string specifying function
            'corex_beta': 1,
            'output': -1, # output is data layer (e.g. hierarchical corex may specify index in decoder as other layer)
            'method': 'mixture',
            'relation': 'equal',
            'constraint': None,
            'recon': None,
            'from_layer': [], # "stats", "recon", in order of list (stats = z_mean, z_var ... recon = x_true, x_pred)
            'from_output': [], 
            'loss_kwargs': {}
        }
        args.update(kwargs) # read kwargs into dictionary, set attributes beyond defaults
        for key in args.keys():
            setattr(self, key, args[key])

        self.beta = beta 

    def set_beta(self, beta):
        self.beta = beta

    def get_inputs(self):
        return self.from_layer

    def get_dict(self):
        print(vars(self))
        return {'type': self.type, 'layer': self.layer, 'encoder': self.encoder, 'weight':self.weight,
                'output': self.output, 'method': self.method, 'recon': self.recon, 'from_layer': self.from_layer, 'from_output': self.from_output}
    # stats takes 
    def make_function(self):
        # interpret type, taking stats and/or recon as inputs 
        function = self._type_to_function()
        return function

    def describe_inputs(self):
        try:
            return self.from_layer, self.from_output
        except:
            _ = self._type_to_function()   # type to function sets from_layer/output to set inputs 
            return self.from_layer, self.from_output

    def get_loss_weight(self):
        return self._get_loss_weight()

    def _type_to_function(self):
        name_suffix = '_'+str(self.layer) if self.layer != -1 else ('_z' if self.encoder else '_loss')
        print("SELF.TYPE ", self.type, type(self.type))
        self.from_addl = []
        # *** RECON *** 
        if self.type in RECON_LOSSES:
            if 'iwae' in self.type:
                self.encoder = True
                self.from_layer = ['stat', 'act']
                self.from_output = ['act']
                ln = 'mul' in self.type
                self.loss_kwargs.update({'recon': self.recon, 'lognormal': ln})
                return Lambda(l.iwae_loss, arguments = self.loss_kwargs, name = 'iwae'+name_suffix)
                # IS IWAE LOSS SAME FOR ADDITIVE / MULTIPLICATIVE?
            elif self.type == 'bce' or self.type == 'binary_crossentropy' or self.type == 'binary_cross_entropy':
                self.from_output = ['act']    
                return Lambda(l.binary_crossentropy, name = 'bce'+name_suffix)
            elif self.type == 'mse' or self.type == 'mean_square_error' or self.type == 'mean_squared_error':
                self.from_output = ['act']    
                return Lambda(l.mean_squared_error, name = 'mse'+name_suffix)

        # *** VAE ***
        elif self.type == 'vae':
            self.from_layer = ['stat']
            return Lambda(l.gaussian_prior_kl, name = 'vae'+name_suffix)

        # *** INFO DROPOUT ***
        elif self.type == 'ido' or self.type == 'info_dropout':        
            self.from_layer = ['stat']
            return Lambda(l.gaussian_prior_kl, name = 'ido'+name_suffix)
        
        # *** COREX ***
        elif 'corex' in self.type:
            if self.method == "mixture" or self.method == 'mog':
                mi = "mixture"
                marginals = not ('joint' in self.type or 'ib' in self.type) # default true => MARGINAL regularization
            else:
                mi = "gaussian"
                marginals = True

            self.from_layer = ['stat']
            self.from_output = ['act']
            self.loss_kwargs.update({'recon': self._parse_recon(), 'mi': mi, 'marginals': marginals, 'beta': self.corex_beta})
            return Lambda(l.corex, arguments = self.loss_kwargs, name = 'corex'+name_suffix)

        # *** TC ***
        elif self.type == 'tc':
            # ASSUMES CONDITIONAL FACTORIZES: to do, incorporate ECHO
            if self.method == 'mixture' or self.method == 'mog':
                try:
                    self.loss_kwargs.update({'maximize': self.loss_weight < 0})
                except:
                    pass
                
                self.from_layer = ['stat']
                return Lambda(m.mixture_tc_gaussian, arguments = self.loss_kwargs, name ='tc_mixture'+name_suffix)
            
            elif self.method == 'discriminator' or self.method == 'gan':
                raise NotImplementedError('TC discriminator not implemented')
            
            elif self.method == 'mcmc' or self.method == 'sampling':
                self.from_layer = ['stat']
                return Lambda(l.tc_sampling, name = 'tc_sampling'+name_suffix)
                #sampling naive
 
        # *** MUTUAL INFORMATION I(X:Z) ***         
        elif self.type == 'mi_joint' or self.type == 'joint_mi':
            # ASSUMES CONDITIONAL FACTORIZES: to do, incorporate ECHO
            if self.method == 'mixture' or self.method == 'mog':
                self.from_layer = ['stat']
                self.loss_kwargs.update({'marginals': False})
                return Lambda(m.mixture_mi_gaussian, arguments = self.loss_kwargs, name = 'mi_joint'+name_suffix)
            elif 'sampling' in self.method or 'mcmc' in self.method:
                self.from_layer = ['stat']
                self.loss_kwargs.update({'marginals': False})
                return Lambda(l.mi_sampling, arguments = self.loss_kwargs, name = 'mi_sampling'+name_suffix)
            else:
                raise NotImplementedError("only sampling and mixture of gaussian estimation of MI is supported.  other ideas?")

        # *** MUTUAL INFORMATION I(X:Zj) ***
        elif self.type == 'mi_marginals' or self.type == 'mi_marginal' or self.type == 'marginal_mi':
            # ASSUMES CONDITIONAL FACTORIZES: to do, incorporate ECHO
            if self.method == 'mixture' or self.method == 'mog':
                self.from_layer = ['stat']
                self.loss_kwargs.update({'marginals': True})
                return Lambda(m.mixture_mi_gaussian, arguments = self.loss_kwargs, name = 'mi_marginals'+name_suffix)
            elif 'sampling' in self.method or 'mcmc' in self.method:
                self.from_layer = ['stat']
                self.loss_kwargs.update({'marginals': True})
                return Lambda(l.mi_sampling, arguments = self.loss_kwargs, name = 'mi_sampling'+name_suffix)
            else:
                raise NotImplementedError("only sampling and mixture of gaussian estimation of MI is supported.  other ideas?")

        elif "echo" in self.type:
            self.from_layer = ['addl']
            return Lambda(l.echo_loss, arguments = self.loss_kwargs, name = 'mi_echo'+name_suffix)


        elif self.type == 'mmd':
            self.from_layer = ['act']
            self.from_output = [] if (self.method == 'prior' or self.method == 'mixture') else ['act']
            return Lambda(l.mmd_loss, arguments = self.loss_kwargs, name = 'mmd'+name_suffix)

        else:
            # TRY IMPORT OF FUNCTION FROM LOSSES
            try:
                func = self._import_loss(self.type, 'keras.losses')
            except:
                func = self._import_loss(self.type, 'l')
            return Lambda(func, name = str(self.type+name_suffix))              


    def _parse_recon(self):
        if self.recon == 'mse' or self.recon == 'mean_squared_error' or self.recon == 'mean_square_error':
            return l.mean_squared_error
        elif self.recon == 'bce' or self.recon == 'binary_crossentropy' or self.recon == 'binary_cross_entropy':
            return l.binary_crossentropy
        else:
            try:
                self._import_loss(self.recon, 'keras.losses')
            except:
                self._import_loss(self.recon, 'l')


    def _import_loss(self, loss, module):
            try:
                mod = importlib.import_module(module)
                mod = getattr(mod, loss)
                print('loss function imported: ', mod, ' from ', module)
                return mod
            except:
                raise ValueError("Cannot import ", loss, " from ", module, '.  Please feed a valid loss function import or argument')


    def _get_loss_weight(self):
        if isinstance(self.weight, str):
            try:
                mod = importlib.import_module(LOSS_WEIGHTS)
                mod = getattr(mod, self.weight)
                loss_weight = mod(self.beta)
            except Exception as e:
                print(e) 
                raise ValueError("Cannot find weight loss function")
        elif isinstance(self.weight, float) or isinstance(self.weight, int) or ininstance(self.weight,  (tf.Variable, tf.Tensor)):
            loss_weight = self.weight
        else:
            raise ValueError("cannot interpret loss weight: ", self.weight) 
        return loss_weight




