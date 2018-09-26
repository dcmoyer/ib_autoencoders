import numpy as np
import keras.backend as K
import importlib
from keras.layers import Dense, Lambda, Reshape
import layers
import tensorflow as tf

class Layer(object):
    def __init__(self, **kwargs):
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
        args = {'layer': -1,
            'latent_dim': None,
            'encoder': True,
            'type': 'add', # module path to import as layer class
            'k': 1,
            'add_loss': True,
            'activation': None, # custom activation (can be an import: module.___)
            #'output': -1, # output is data layer (e.g. hierarchical corex may specify index in decoder as other layer)
            'density_estimator': None,
            'layer_kwargs': {}
                # kw args for Keras or other layer
        }
        args.update(kwargs) # read kwargs into dictionary, set attributes beyond defaults
        for key in args.keys():
            setattr(self, key, args[key])

        self.try_activations()

    def equals(self, other_args):
        return self.__eq__(other_args)

    def try_activations(self, kw = False):
        try:
            act = self.layer_kwargs['activation'] if kw else self.activation
            actstr = str(act)
        
            mod = importlib.import_module('keras.activations')
            act = getattr(mod, actstr)
        except:
            try:
                print()
                print("***")
                print("Trying split import ", ".".join(actstr.split(".")[:-1]), actstr.split(".")[-1])
                print("***")
                mod = importlib.import_module(".".join(actstr.split(".")[:-1]))
                act = getattr(mod, actstr.split(".")[-1])
            except:
                #print("Couldn't import ", actstr.split(".")[-1])
                pass

        if not self.layer_kwargs.get('activation', True):
            self.layer_kwargs['activation'] = act
        

    def make_function_list(self, index = 0):
        stats_list = []
        act_list = []
        addl_list = []
        #latent_dim = latent_dim if latent_dim is not None else self.latent_dim

        if self.type == 'echo' and not self.layer_kwargs.get('d_max', False) and self.k != 1:
            self.layer_kwargs['d_max'] = self.k
            # echo params fed via kwargs, otherwise defaults (i.e. k = 1, no d_max => default)
            self.k == 1 # ensures only one z_activation layer
            

        for samp in range(self.k):
            net = 'Enc' if self.encoder else 'Dec'
            name_suffix = str(net)+'_'+str(index)+'_'+str(samp) if self.k > 1 else str(net)+'_'+str(index)
            if self.type in ['add', 'vae', 'additive']: 
                if samp == 0:
                    z_mean = Dense(self.latent_dim, activation='linear',
                               name='z_mean'+name_suffix, **self.layer_kwargs)
                    z_logvar = Dense(self.latent_dim, activation='linear',
                               name='z_var'+name_suffix, **self.layer_kwargs)            
                    stats_list.append([z_mean, z_logvar])
                # NOTE: using layer_kwargs for dense... samplers don't need any here
                z_act = Lambda(layers.vae_sample, name = 'z_act_'+name_suffix) #arguments = self.layer_kwargs)
                act_list.append(z_act)

            elif self.type in ['mul', 'ido', 'multiplicative']:
                if samp == 0:   
                    z_mean = Dense(self.latent_dim, activation='linear', name='z_mean'+name_suffix, **self.layer_kwargs)
                    z_logvar = Dense(self.latent_dim, activation='linear', name='z_var'+name_suffix, **self.layer_kwargs)
                    stats_list.append([z_mean, z_logvar])
                z_act = Lambda(layers.ido_sample, name = 'z_act_'+name_suffix, arguments =self.layer_kwargs)
                act_list.append(z_act)

            elif self.type in ['echo']:
                if samp == 0:   
                    # slightly different here... layer_kwargs used for echo / lambda
                    z_mean = Dense(self.latent_dim, activation='linear', name='z_mean'+name_suffix)# **self.layer_kwargs)
                    z_act = layers.Echo(name = 'echo_act'+name_suffix, **self.layer_kwargs)
                    z_act.trainable = self.layer_kwargs['trainable']
                    print("Trainable ? ", z_act.trainable, self.layer_kwargs['trainable'], z_act)
                    #z_act = Lambda(layers.echo_sample, name = 'z_act_'+name_suffix, arguments = self.layer_kwargs)
                    capacity = Lambda(z_act.get_capacity, name = 'capacity'+name_suffix)
                    #capacity = Lambda(layers.echo_capacity, name = 'capacity'+name_suffix, arguments = {'init': self.layer_kwargs['init']})#, 'batch': self.layer_kwargs['batch']})
                # note: k = 1 if k used as d_max, otherwise will have k separate layer calls
                # tf.get_variable  self.layer_kwargs['init']
                stats_list.append([z_mean])
                act_list.append(z_act)
                addl_list.append(capacity)

            elif self.type in ['bir', 'constant_additive']:
                if samp == 0:   
                    z_mean = Dense(self.latent_dim, activation='linear', name='z_mean'+name_suffix)
                    mi = self.layer_kwargs.get('mi', None)
                    if mi is None:
                        var = self.layer_kwargs.get('variance', self.layer_kwargs.get('sigma', self.layer_kwargs.get('noise', None))) 
                        if var is None:
                            raise ValueError("Please enter layer_kwarg['mi'] or ['variance'] for bounded rate autoencoder")
                    else:
                        var = -2*mi/self.latent_dim
                    self.layer_kwargs['variance'] = var
                    z_logvar = Lambda(layers.constant_layer, name = 'z_var'+name_suffix, arguments = {'variance':self.layer_kwargs['variance'], 'batch': self.layer_kwargs['batch']})
                    #K.constant(np.log(var), shape = (self.latent_dim,), name = 'z_var'+name_suffix)
                    #Dense(self.latent_dim, activation='linear', name='z_var'+name_suffix, **self.layer_kwargs)
                    stats_list.append([z_mean, z_logvar])
                z_act = Lambda(layers.vae_sample, name = 'z_act_'+name_suffix)
                act_list.append(z_act)

            elif self.type in ['autoregressive_flow', 'af', 'ar_flow', 'arf']:
                #if samp == 0:
                flow = AR_flow(name = 'ar_flow'+name_suffix)
                density = Lambda(flow.get_density, name = 'density'+name_suffix)
                act_list.append(flow)
                addl_list.append(density)

            elif self.type in ['iaf', 'inverse_flow']:
                z_mean = Dense(self.latent_dim, activation='linear',
                              name='z_mean'+name_suffix)#**self.layer_kwargs)
                z_logvar = Dense(self.latent_dim, activation='linear',
                              name='z_var'+name_suffix)# **self.layer_kwargs)            
                stats_list.append([z_mean, z_logvar])

                
                self.try_activations(kw = True)
                    

                iaf = layers.IAF(name = 'z_act'+name_suffix,
                    **self.layer_kwargs)
                iaf_density = Lambda(iaf.get_density, name = 'iaf_conditional'+name_suffix)
                #print('*** CONSTANT JACOBIAN IAF ???? ***', iaf.is_constant_jacobian)
                
                act_list.append(iaf)
                addl_list.append(iaf_density) # doesn't matter what it calls?  sample to be safe

            elif self.type in ['maf', 'masked_arf']:#, 'gaussian_af', 'gaussian_arf', 'gaussian_maf']:
                tf_made = True
                if tf_made:
                    # args['steps']= self.layer_kwargs.get('steps', 1)
                    # args['layers'] = self.layer_kwargs.get('layers', 1)
                    # args['mean_only'] = self.layer_kwargs.get('mean_only', 1)
                    # args['activation'] = self.layer_kwargs.get('activation', 'relu')
                    
                    self.try_activations(kw = True)
                    print(name_suffix)
                    self.layer_kwargs["name"] = 'maf_density'+str(name_suffix)
                    reshape = Reshape((self.latent_dim,), name = 'conv_reshape'+name_suffix)

                    #if len(K.int_shape(z_mean)) > 2:
                        # tf.reduce_prod(tf.shape(z_mean)[1:])
                    #   z_mean = Reshape([-1, dim])(z_mean) 
                        #z_mean = Reshape([-1, *z_mean._keras_shape[1:]])(z_mean) 
                        #z_mean = Flatten()(z_mean)
                    
                    maf = Lambda(layers.tf_masked_flow, arguments = self.layer_kwargs)#, name = 'masked_flow'+name_suffix)
                    
                    stats_list.append([reshape])
                    act_list.append(maf)
                    # directly gives log probability! (from input z)

                else:
                    made_network = layers.MADE_network(steps = self.layer_kwargs.get('steps', 1), 
                                                    layers = self.layer_kwargs.get('layers', 1),
                                                    mean_only = self.layer_kwargs.get('mean_only', 1),
                                                    name = 'made_network'+name_suffix)
                    made_jacobian = Lambda(made_network.get_log_det_jac, name = 'made_jac'+name_suffix)
                    # prob z, to be fed to loss (treat this as recon?)
                    act_list.append(made_network)
                    addl_list.append(made_jacobian)

            else:
                # import layer module by string (can be specified either in activation or layer_kwargs)
                try:
                    #self.try_activations()
                    spl = str(self.activation).split('.')
                    if len(spl) > 1:
                        path = '.'.join(spl[:-1])
                        mod = importlib.import_module(path)
                        mod = getattr(mod, spl[-1])
                        self.layer_kwargs['activation'] = mod
                    else:
                        if not self.layer_kwargs.get('activation', True):
                            self.layer_kwargs['activation'] = self.activation
                except Exception as e:
                    print()
                    print("trying activationfor layer ", self.type)
                    print(e) #raise NotImplementedError("Coding error in importing activation.  Specify as Keras activation str or module.function")
                    print()

                try:
                    self.try_activations(kw = True)
                except Exception as e:
                    print()
                    print("trying layer kw activation for ", self.type)
                    print(e) #raise NotImplementedError("Coding error in importing activation.  Specify as Keras activation str or module.function")
                    print()

                try:
                    spl = str(self.type).split('.')
                    if len(spl) > 1:
                        path = '.'.join(spl[:-1])
                        mod = importlib.import_module(path)
                        self.layer_kwargs['name'] = str(path+name_suffix)
                        mod = getattr(mod, spl[-1])
                        z_act = mod(self.latent_dim, **self.layer_kwargs)
                    else:
                        mod = importlib.import_module(str('keras.layers'))
                        #mod = importlib.import_module(str('tensorflow.python.keras.layers'))
                        self.layer_kwargs['name'] = str(self.type + name_suffix)
                        
                        if self.type == 'Dense':
                            z_act = Dense(self.latent_dim, **self.layer_kwargs)
                        else:
                            z_act = getattr(mod, self.type)
                            z_act = z_act(self.latent_dim, **self.layer_kwargs)
                            
                except:
                    raise AttributeError("Error Importing Activation ", self.type)
                act_list.append(z_act)
       
        return {'stat': stats_list, 'act': act_list, 'addl': addl_list}
            # if stats_list:
            #     return [stats_list, act_list]
            # else:
            #     return [act_list]



def default_noise_layer(str_name, latent_dim):
    k = 1
    if str_name == 'vae':
        # add loss by default if vae, else just add noise layer
        add_loss = 'vae' if str_name == 'vae' else False
        str_name = 'add'
    elif str_name == 'ido' or str_name == 'mul':
        # add loss by default if info dropout, else just add noise layer
        add_loss = 'ido' if str_name == 'ido' else False
        str_name = 'mul'
    elif str_name == 'iwae':
        str_name = 'add'
        k = 5
        add_loss = 'iwae'
    else:
        raise ValueError("Please specify noise_layer as one of 'vae', 'add', 'ido', 'mul', 'iwae' or via a list of layer argument dictionaries.")

    return Layer(** {"layer":-1, "latent_dim": latent_dim, "type": str_name, "k": 1, "add_loss": add_loss})