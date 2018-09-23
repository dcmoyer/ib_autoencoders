import numpy as np
import keras.backend as K
import importlib
import tensorflow as tf
import losses as l
import model
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
        #name_suffix = '_'+str(self.layer) if self.layer != -1 else ('_z' if self.encoder else '')
        name_suffix = '_'+('recon' if self.type in RECON_LOSSES else ('reg' if self.weight != 0 else ''))+('_'+str(self.layer) if self.layer != -1 else '')
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

        elif self.type == "echo":# in self.type:
            self.from_layer = ['addl'] #['addl']
            return Lambda(l.echo_loss, arguments = self.loss_kwargs, name = 'mi_echo'+name_suffix)

        elif self.type == "echo_var":# in self.type:
            self.from_layer = ['addl']
            return Lambda(l.echo_var, arguments = self.loss_kwargs, name = 'echo_var'+name_suffix)

        elif self.type == "echo_min":# in self.type:
            self.from_layer = ['addl']
            return Lambda(l.echo_minmax, arguments = {"_min": True}, name = 'echo_min'+name_suffix)

        elif self.type == "echo_max":# in self.type:
            self.from_layer = ['addl']
            return Lambda(l.echo_minmax, arguments = {"_min": False}, name = 'echo_max'+name_suffix)

        elif self.type == 'bir':
            self.from_layer = ['stat']
            return Lambda(l.bir, name = 'bir_mi'+name_suffix)

        elif self.type == 'mmd':
            self.from_layer = ['act']
            self.from_output = [] if (self.method == 'prior' or self.method == 'mixture') else ['act']
            return Lambda(l.mmd_loss, arguments = self.loss_kwargs, name = 'mmd'+name_suffix)

        elif self.type in ['iaf', 'iaf_encoder', 'iaf_conditional', 'iaf_density']:
            self.from_layer = ['addl']
            return Lambda(l.dim_sum_one, arguments = {"keepdims": True}, name = 'iaf'+name_suffix)

        elif self.type in ['made_density', 'made_marginal']:
            prev_made = False
            if not prev_made:
                self.from_layer = ['act'] #arguments = {"keepdims":True}
                return Lambda(l.dim_sum_one, arguments = {"keepdims": True}, name = 'made'+name_suffix)
                #return Lambda(l.identity_one, name = 'made'+name_suffix)
            else:
                # default is gaussian_inputs (i.e. sample isotropic gauss, transform into mean / mean + std of gaussian)
                # specify loss['mean_only'] = false if stddev learned, otherwise mean_only w/ stddev = 1
                self.type = 'made_density'
                #
                # RANDOM NOISE =?  Q(z) ESTIMATE .... INPUT = Z ACT as points to evaluate (PROVIDES SHAPE)
                # loss function can take E_q r(z), but also E_q q(z|x) (GAUSSIAN for now, later IAF)
                
                # treat as a RECON loss
                self.from_layer = ['act', 'addl']
                return Lambda(l.made_marginal_density, arguments = self.loss_kwargs, name = 'made'+name_suffix)

        # attempt at mismatch sampling from p(z)?
        elif self.type in ['made_decoder_density', 'prior_distortion', 'prior_vim', 'prior_relevant']:
            # TRY TO GET RECON UNDER P(Z) MEANS (OR rand draws from conditional).. sample from MAF?
            self.type = 'made_decoder_density'
            self.from_layer['act']
            return Lambda(model._decoder, name = 'prior_recon'+name_suffix)

        else:
            # TRY IMPORT OF FUNCTION FROM LOSSES
            try:
                func = self._import_loss(self.type, 'keras.losses')
            except:
                print("Trying import ", self.type, " from l ")
                func = self._import_loss(self.type, 'losses')
            self.from_layer = ['stat']
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
                print("Trying import ", self.recon, " from l ")
                self._import_loss(self.recon, 'losses')


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




