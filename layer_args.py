import numpy as np
import keras.backend as K
import importlib
from keras.layers import Dense, Lambda
import layers


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
            'layer_kwargs': {}
                # kw args for Keras or other layer
        }
        args.update(kwargs) # read kwargs into dictionary, set attributes beyond defaults
        for key in args.keys():
            setattr(self, key, args[key])

    def equals(self, other_args):
        return self.__eq__(other_args)

    def make_function_list(self, index = 0):
        stats_list = []
        act_list = []
        #latent_dim = latent_dim if latent_dim is not None else self.latent_dim

        for samp in range(self.k):
            net = 'E' if self.encoder else 'D'
            name_suffix = str(net)+'_'+str(index)+'_'+str(samp) if self.k > 1 else str(net)+'_'+str(index)
            if self.type in ['add', 'vae']: 
                if samp == 0:
                    z_mean = Dense(self.latent_dim, activation='linear',
                               name='z_mean'+name_suffix, **self.layer_kwargs)
                    z_logvar = Dense(self.latent_dim, activation='linear',
                               name='z_var'+name_suffix, **self.layer_kwargs)            
                    stats_list.append([z_mean, z_logvar])
                z_act = Lambda(layers.vae_sample, name = 'z_act'+name_suffix, **self.layer_kwargs)
                act_list.append(z_act)

            elif self.type in ['mul', 'ido']:
                if samp == 0:   
                    z_mean = Dense(self.latent_dim, activation='linear', name='z_mean'+name_suffix, **self.layer_kwargs)
                    z_logvar = Dense(self.latent_dim, activation='linear', name='z_var'+name_suffix, **self.layer_kwargs)
                    stats_list.append([z_mean, z_logvar])
                z_act = Lambda(layers.ido_sample, name = 'z_act'+name_suffix, **self.layer_kwargs)
                act_list.append(z_act)

            else:
                # import layer module by string (can be specified either in activation or layer_kwargs)
                try:
                    if self.activation is None and self.layer_kwargs.get('activation', False):
                        self.activation = self.layer_kwargs['activation']
                    spl = str(self.activation).split('.')
                    if len(spl) > 1:
                        path = '.'.join(spl[:-1])
                        mod = importlib.import_module(path)
                        mod = getattr(mod, spl[-1])
                        self.layer_kwargs['activation'] = mod
                    else:
                        if not self.layer_kwargs.get('activation', True):
                            self.layer_kwargs['activation'] = self.activation
                except:
                    raise NotImplementedError("Coding error in importing activation.  Specify as Keras activation str or module.function")

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
                        self.layer_kwargs['name'] = str(self.type + name_suffix)
                        
                        if self.type == 'Dense':
                            z_act = Dense(self.latent_dim, **self.layer_kwargs)
                        else:
                            z_act = getattr(mod, self.type)
                            z_act = z_act(self.latent_dim, **self.layer_kwargs)
                            
                except:
                    raise AttributeError("Cannot import layer module.  Set 'type' to 'add', 'mul', or import path.")
                act_list.append(z_act)
       
        return {'stat': stats_list, 'act': act_list}
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