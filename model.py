import numpy as np
import keras.backend as K
import tensorflow as tf
import importlib
import json
from collections import defaultdict
from keras import backend as K
from keras.layers import Input, Dense, merge, Lambda #Concatenate, 
from keras.layers import Activation, BatchNormalization, Lambda, Reshape
from keras.callbacks import Callback, TensorBoard, LearningRateScheduler
import keras.models
import keras.layers
import keras.optimizers
import keras.initializers
import utils
import layer_args
from loss_args import Loss
import dataset
import losses as l
import analysis

K.set_image_dim_ordering('tf')
RECON_LOSSES = ['bce', 'mse', 'binary_crossentropy', 'mean_square_error', 'mean_squared_error', 'iwae']

class Model(object):
    def __init__(self, dataset, *args, **kwargs):
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__(key, hyper_params[key])

class WrapModel(Model):
    def __init__(self, model):
        if not callable(getattr(model, 'fit', None)):
            pass # wrap fit
        if not callable(getattr(model, 'transform', None)):
            pass # wrap transf
        if not callable(getattr(model, 'predict', None)):
            pass # wrap predict
        if not callable(getattr(model, 'generate', None)):
            pass # wrap generate

# TRANSFORM INTO Load from Config METHOD
# static methods creating, loading, saving, adding attributes


class NoiseModel(Model):
    def __init__(self, dataset, args_dict = {}, config = None, dflt = {}):
        # dflt as dictionary (set by session)
        #if dflt is None:
        #   args = utils.load_from_config('configs/dflt.json')
        # All failed dictionary reads first check default config, then fall back to given value

        self.args = {
            'dataset': 'mnist', # specify + import dataset class
            'epochs': 100,
            'batch': 100,
            'optimizer': 'Adam',
            'initializer': 'glorot_uniform',
            'optimizer_params': {},
            'lr': 0.001,
            'input_shape': None,
            'activation': {'encoder': 'softplus', 'decoder': 'softplus'},
            'output_activation': 'sigmoid',
            'encoder_dims': None, #[200, 200, 50],
            'decoder_dims': None,
            'noise_layer': 'vae',
            'layers': None,
            #'intermediate_layers': None
            'recon': None,
            #'recon_layers': None,
            'losses': None,
            'beta': 1.0,
            'anneal_schedule': None,
            'anneal_function': None,
        }
        if config is not None:
            self.args.update(json.load(open(config)))

        # read kwargs into dictionary
        self.args.update(args_dict) 
        for key in self.args.keys():
            setattr(self, key, self.args[key])

        self._parse_args()
        self._parse_layers_and_losses()

        # initialize dictionary (with keys = layer index) of dict of called layers (keys = 'stat', 'act')
        self.encoder_layers = [] #defaultdict(dict)
        self.decoder_layers = [] #defaultdict(dict)

    def _parse_args(self):
        
        if self.dataset == 'mnist':
            self.dataset = dataset.MNIST(binary = False)
        elif self.dataset == 'binary_mnist':
            self.dataset = dataset.MNIST(binary = True)
        elif self.dataset == 'omniglot':
            pass
        elif self.dataset == 'celeb_a':
            pass
        elif self.dataset == 'dsprites':
            self.dataset = dataset.DSprites()
        # Training Params
        #self.epochs = args.get('epochs', dflt.get('epochs', 100))
        #self.batch = args.get('batch', dflt.get('batch', 100))
        #optimizer = args.get('optimizer', dflt.get('optimizer', 'Adam'))
        #initializer = args.get('initializer', dflt.get('initializer', 'glorot_uniform'))
        #params = args.get('optimizer_params', dflt.get('latent_dims', {}))
        # feed args to keras optimizers
        self.optimizer = getattr(keras.optimizers, self.optimizer)(**self.optimizer_params)
        #self.initializer = getattr(keras.initializers, self.initializer)

        self.lr_callback = False
        if isinstance(self.lr, str):
            try:
                mod = importlib.import_module(str('lr_sched'))
                # LR Callback will be True /// self.lr = function of epochs -> lr
                self.lr = getattr(mod, self.lr)
                self.lr_callback = isinstance(self.lr, str)
            except:
                self.lr = dflt.get('lr', .001)
                raise Warning("Cannot find LR Schedule function.  Proceeding with default, constant learning rate.")    


        # self.lr = args.get('lr', dflt.get('latent_dims', .001)) 
        # try:
        #   self.lr = getattr(lr_sched, self.lr) if isinstance(self.lr, str) else self.lr # str or float
        # except:
        #   self.lr = dflt.get('latent_dims', .001)
        #   raise Warning("Cannot find LR Schedule function.  Proceeding with default, constant learning rate.")
        # self.lr_callback = isinstance(self.lr, str)
        
        # Architecture Args
        if self.encoder_dims is None:
            try:
                self.encoder_dims = self.latent_dims
            except Exception as e:
                print(e)
                raise ValueError

        if self.decoder_dims is None: # check if has decoder_dims arg
            self.decoder_dims = list(reversed(self.encoder_dims[:-1]))
            self.decoder_dims.append(self.dataset.dim)
        else:
            if self.decoder_dims[-1] != self.dataset.dim:
                self.decoder_dims.append(self.dataset.dim)
            print("Using custom decoder architecture")
        #   self.decoder_dims = args.get('decoder_dims')


        # if noise layers not specified, get default based on noise_layer
        if not self.layers:
            self.layers = None
        if self.layers is None:
            # default noise layer for 'add', 'mul', 'vae', 'ido'
            self.layers = [layer_args.default_noise_layer(self.noise_layer, self.encoder_dims[-1])]

        # *** Check for conflicts in arguments (later) ***
        # import recon function?
        
        #self.recon = [self.recon] if isinstance(self.recon, str) else self.recon
        #self.reg = args.get('reg', dflt.get('reg', None))
        #self.recon_layers = args.get('recon_layers', dflt.get('recon_layers', None)) # or []?
        #self.reg = [self.reg] if isinstance(self.reg, str) else self.reg
        
        # Corex Loss function must specify recon loss, not recon_layers here

        #self.reg_layers = args.get('reg_layers', dflt.get('reg_layers', [-1])) #regularize last layer
        #self.reg_layers = [self.reg_layers] if isinstance(self.reg_layers, int) else self.reg_layers

        #self.reg_weights = args.get('reg_weights', dflt.get('reg_weights', [1]))
        #if isinstance(self.reg_weights, float) or isinstance(self.reg_weights, str) or isinstance(self.reg_weights, int):  
        #    self.reg_weights = [self.reg_weights]  

        # if anneal_function, elif sched, else beta static
        #self.beta = args.get('reg_weights', dflt.get('reg_weights', 1.0))
        #self.anneal_schedule = args.get('anneal_schedule', dflt.get('anneal_schedule', None))
        if self.anneal_schedule and not isinstance(self.beta, list):
            self.anneal_schedule = None
            raise Warning('For anneal schedule, Beta should be a list of lists, with each list specifying changepoints of Lagrange multiplier).  Proceeding with static Beta')
        if self.anneal_schedule is not None and not isinstance(self.anneal_schedule, list):
            raise TypeError('Anneal schedule should be a list')
        if isinstance(self.anneal_schedule, list) and isinstance(self.beta, list) and len(self.anneal_schedule)!= len(self.beta):
            raise ValueError('Anneal schedule and Beta list must be same length (Note: you should place a 0 in anneal_schedule to correspond to first Beta.')

        #self.anneal_function = args.get('anneal_function', dflt.get('anneal_function', None))
        self.anneal = (self.anneal_schedule is not None or self.anneal_function is not None)

        if isinstance(self.recon, list):
            self.recon = str(self.recon[0])

    def _parse_layers_and_losses(self):
        # list of indices of regularized or custom layers
        self._enc_latent_ind = []
        #self._enc_ind_args = []
        self._dec_latent_ind = []
        #self._dec_ind_args = []
        #self._encoder_losses = []
        #self._decoder_losses = []
        self._enc_loss_ind = []
        self._dec_loss_ind = []
        print(self.layers)

        # loop through to record which layers have losses attached
        if self.losses is not None and isinstance(self.losses, list):
            for i in range(len(self.losses)):
                lossargs = self.losses[i]
                if lossargs.get('encoder', True) and (lossargs.get('type') not in RECON_LOSSES and lossargs.get('add_loss') not in RECON_LOSSES): # loss defaults to encoder unless in recon
                    if 'encoder' not in lossargs:
                        print("WARNING: Loss entry ", i, ": ",  lossargs.get('type', '') ," defaulting to encoder")

                    self._enc_loss_ind.append(len(self.encoder_dims)-1 
                                                if lossargs.get('layer', -1) == -1
                                                else lossargs['layer'])
                else:
                    self._dec_loss_ind.append(len(self.decoder_dims)-1 
                                                if lossargs.get('layer', -1) == -1
                                                else lossargs['layer'])
                    if lossargs.get('layer', -1) != -1:
                        print("WARNING: loss on intermediate decoder layers doesn't seem to make sense.  If you want a Corex layer to reconstruct a decoder layer, specify via layer arguments and mapping btwn indices of layer_args list and latent_dims list")
        else:
            self.losses = []
            print("WARNING: Losses not specified as list of dictionaries?  DEFAULTING to Recon argument (+ any default noise layers specified) ")

        # loop through to record which layers have special layer arguments (noise regularization or non-dense layer type)
        for i in range(len(self.layers)):
            layerargs = self.layers[i]
            if layerargs.get('encoder', True):
                if 'encoder' not in layerargs:
                    print("WARNING: Layer entry ", i, " defaulting to encoder")
                # for each entry in layer args list, record corresponding index in latent_dims
                self._enc_latent_ind.append(len(self.encoder_dims)-1 
                                                if layerargs.get('layer', -1) == -1
                                                else layerargs['layer'])
                # index of layer arg in latent_dim specification
                #self._enc_ind_args.append(i)
            else:
                self._dec_latent_ind.append(len(self.decoder_dims)-1
                                                if layerargs.get('layer', -1) == -1
                                                else layerargs['layer']) 
                #self._dec_ind_args.append(i)

            # ADDING DEFAULT "ADD_LOSS" args (same process as above for lossargs): 
            # what happens to IWAE?
            if layerargs.get('add_loss', False):# and :
                if isinstance(layerargs['add_loss'], bool):
                    if layerargs['type'] in ['add', 'vae']:
                        layerargs['type'] = "vae"
                    elif layerargs['type'] in ['mul', 'ido']:
                        layerargs['type'] = "ido"
                    elif "iwae" in layerargs['type']:
                        layerargs['type'] = "iwae"

                if layerargs.get('encoder', True) and layerargs['add_loss'] not in RECON_LOSSES:
                    self._enc_loss_ind.append(len(self.encoder_dims)-1 
                                                if layerargs.get('layer', -1) == -1
                                                else layerargs['layer'])
                else:
                    self._dec_loss_ind.append(len(self.decoder_dims)-1 
                                                if layerargs.get('layer', -1) == -1
                                                else layerargs['layer'])
                    if layerargs.get('layer', -1) != -1:
                        print("WARNING: loss on intermediate decoder layers doesn't seem to make sense.  If you want a Corex layer to reconstruct a decoder layer, specify via layer arguments and mapping btwn indices of layer_args list and latent_dims list")

                addl_loss = Loss(**{'type': layerargs['add_loss'], 
                                    'layer': layerargs.get('layer', -1), 
                                    'encoder': layerargs.get('encoder', True), 
                                    'weight': 1})
                print("APPENDING ", addl_loss.type)
                self.losses.append(addl_loss)
                print("LOSSES LEN ", len(self.losses))

            if not self._dec_loss_ind and self.recon is not None:
                print("EXTRA APPEND")
                self._dec_loss_ind.append(len(self.decoder_dims)-1)
                recon_loss = Loss(**{'type': self.recon,
                                    'layer': -1,
                                    'encoder': False,
                                    'weight': 1
                                })
                self.losses.append(recon_loss)

    def fit(self, x_train, y_train = None, x_val = None, y_val = None):
        x = Input(shape = (self.dataset.dim,)) if self.input_shape is None else Input(shape = self.input_shape) 
        self.recon_true = x# Lambda(lambda y : y, name = 'x_true')(x)
        print('INPUT SHAPE ', self.input_shape, ' x shape ', x, self.input_shape.insert(-1, 0) if self.input_shape is not None else '')
        print("***********************         ENCODER          *****************************")
        self.encoder_layers = self._build_architecture([x], encoder = True)
        print("***********************         DECODER          *****************************")
        self.decoder_layers = self._build_architecture(self.encoder_layers[len(self.encoder_layers)-1]['act'], encoder = False)
        self.model_outputs, self.model_losses, self.model_loss_weights = self._make_losses()
        callbacks = self._make_callbacks()
        print(self.model_outputs)
        print([type(o) for o in self.model_outputs])
        self.model = keras.models.Model(inputs = x, outputs = self.model_outputs)
        print(self.model.summary())
        for i in self.model.layers[1:]:
            try:
                print(i.name, i.activation)
            except:
                pass
        print('outputs ', self.model_outputs)
        print('losses ', self.model_losses)
        self.model.compile(optimizer = self.optimizer, loss = self.model_losses, loss_weights = self.model_loss_weights) # metrics?
        self.hist = self.model.fit(x_train, ([x_train] if y_train is None else [y_train])*len(self.model_outputs), 
                           epochs = self.epochs, batch_size = self.batch, callbacks = callbacks)
        # how to get activation layers?
        examples = x_train[0:10]
        z = self._encoder(x = examples)
        
        #means = K.mean(z, axis = 0)
        #sigs = K.sqrt(K.var(z, axis = 0))
        analysis.plot_traversals(examples, 
                        self._encoder(), 
                        self._decoder(),
                        z_act = z,
                        imgs = 3)#,
                        #means = means,
                        #sigs = sigs)

        #tf_mod(kz, x_out, sess), top = latent_dims[-1], prefix = os.path.join(os.path.dirname(os.path.realpath(__file__)), log_path, '_'), z_act = z_acts, means= means, sigmas = sigs, imgs = p)

    def _encoder(self, x = None):
        for i in self.model.layers:
            if 'z_act' in i.name:
                final_latent = i.name

        get_z = K.function([self.model.layers[0].get_input_at(0)], [
                        self.model.get_layer(final_latent).get_output_at(0)])
        return get_z if x is None else get_z([x])[0]

    def _decoder(self, x = None):
        for i in self.model.layers:
            if 'z_act' in i.name:
                final_latent = i.name

        z_inp = Input(shape = (self.encoder_dims[-1],))
        z_out = z_inp
        call_ = False
        for layr in self.model.layers:
            # only call decoder layer_list
            if call_ and not isinstance(layr, keras.layers.core.Lambda):#and not ('vae' in layr.name or 'noise' in layr.name or 'info_dropout' in layr.name):
                z_out = layr(z_out)
            if layr.name == final_latent:
                call_ = True
            # doesn't work with new naming convention
            #if layr.name == 'decoder' or layr.name == 'ci_decoder':
            #    call_ = False
        generator = keras.models.Model(input = [z_inp], output = [z_out])
        return generator

    def _make_losses(self):
        self.model_outputs = []
        self.model_losses = []
        self.model_loss_weights = []

        
        #for ls in ['dec', 'enc']:
        for i in range(len(self.losses)):
            print('loss type ', self.losses[i])
            #print(vars(self.losses[i]))
            loss = Loss(**self.losses[i]) if isinstance(self.losses[i], dict) else self.losses[i]

            enc = loss.encoder
            #inds = self._enc_loss_ind if enc  else self._dec_loss_ind
            #latent_inds = self._enc_latent_ind if enc else self._dec_latent_ind
            #loss_dict = self._enc_losses if enc else self._dec_losses
                
            
            # remove beta from Loss constructor and check if weight callable 
            
            # loss_list.append({'loss': loss,
            #                 'function': loss.make_function(),
            #                 'weight': loss.get_loss_weight(), 
            #                 'inp_list': loss.describe_inputs()
            #                 })

            fn = loss.make_function()
            print('loss func *********', loss.type, loss.layer)
            lyr_args, output_args = loss.describe_inputs()
            outputs = []
            print('describe inputs', lyr_args, output_args)
            
            for j in range(len(lyr_args)): # 'stat' or 'act'
                # enc / dec already done
                layers = self.encoder_layers if enc else self.decoder_layers
                lyr = loss.layer 
                 
                #  'stat' or 'act' for enc/dec layer # lyr
                if 'act' in lyr_args[j]:# == 'act':
                    #if not loss.encoder and lyr in [-1, len(self.decoder_dims)-1]:
                    #    layers[lyr]['act'].insert(0, self.recon_true)
                    outputs.extend(layers[lyr][lyr_args[j]])
                elif 'stat' in lyr_args[j]:
                    print('adding stat from ', layers[lyr], 'choosing lyr_arg', lyr_args[j])
                    outputs.extend(layers[lyr][lyr_args[j]][0])
            print('outputs for layer ', outputs)
            for j in range(len(output_args)):
                layers = self.decoder_layers
                lyr = loss.output
                if 'act' in output_args[j]:# == 'act':
                    # all output activations get either recon_true or encoder activation (for corex)
                    if (lyr == -1): #[-1, len(self.decoder_dims)-1]):
                        recon_true = self.recon_true
                    else:
                        if len(self.encoder_layers[lyr]['act']) == 1:
                            recon_true = layers[lyr]['act'][0]
                            print('rECON TrUE ', recon_true) 
                        else:
                            raise NotImplementedError("Cannot handle > 1 activation for intermediate layer reconstruction")
                    
                    layers[lyr][output_args[j]].insert(0, recon_true)
                    #print('act output', layers[lyr][output_args[j]])
                    outputs.extend(layers[lyr][output_args[j]])
                elif 'stat' in output_args[j]:
                    #print('stat output', layers[lyr][output_args[j]])
                    outputs.extend(layers[lyr][output_args[j]][0])
            print('outputs for layer ', outputs)
        
            self.model_outputs.append(fn(outputs))
            self.model_losses.append(l.dim_sum)
            self.model_loss_weights.append(loss.get_loss_weight())
            print("OUTPUTS ", self.model_outputs)
            print("Losses: ", self.model_losses)
            
        # losses are all Lambda layers on tensors to facilitate optimization in tensorflow
            # return dimension-wise loss and then sum in dummy loss function?

        # allow returning dimension-wise loss for plot_traversals, e.g.  
        # mean and sum over batch / dimensions should be done here

        return self.model_outputs, self.model_losses, self.model_loss_weights

    def _make_callbacks(self):
        callbacks = []
        if self.lr_callback:
            callbacks.append(LearningRateScheduler(self.lr))
        return callbacks

    def transform(self):
        pass

    def predict(self, inp):
        def my_predict(model, data, layer_name, multiple = True):
            func = K.function([model.layers[0].get_input_at(0)],
                        [model.get_layer(layer_name).get_output_at(0)])
            return func([data])[0]
        try:
            return my_predict(self.model, inp if inp is not None else self.dataset.x_train, 'decoder')
        except:
            print("my_predict failed (line 327 of model.py")
            return self.model.predict(inp if inp is not None else self.dataset.x_train)
        #raise NotImplementedError
        

    def generate(self):
        pass

    def _build_architecture(self, input_list = None, encoder = True):
        # if in layer args... make_function_list
        # else, Dense

        # assignments intended to make function & assigments general for encoder / decoder
        if encoder:
            #self.encoder_layers = defaultdict(list)
            layers_list = self.encoder_layers
            dims = self.encoder_dims
            ind_latent = self._enc_latent_ind
            #ind_args = self._enc_ind_args
        else:
            #self.decoder_layers = defaultdict(list)
            layers_list = self.decoder_layers
            dims = self.decoder_dims
            ind_latent = self._dec_latent_ind
            #ind_args = self._dec_ind_args
        print()
        for layer_ind in range(len(dims)):
            #if layer_ind == len(dims):
                # if prev layer is convolutional


            layers_list.append(defaultdict(list))

            if layer_ind == 0:
                if input_list is not None:
                    current_call = input_list
                else:
                    raise ValueError("Feed input tensor to _build_architecture")
            else:
                # list of inputs ([] to start)
                current_call = layers_list[layer_ind-1]['act']
                

            if layer_ind in ind_latent:
                # retrieve layer_arguments index for given encoding layer
                arg_ind = ind_latent.index(layer_ind) 
                if layers_list[arg_ind].get('latent_dim', None) is None:
                    layers_list[arg_ind]['latent_dim'] = dims[layer_ind]
                print("ADDING DIM ", dims[layer_ind], layer_ind, ind_latent)
                print(layers_list[arg_ind])
                layer = layer_args.Layer(** layers_list[arg_ind])
            else:
                # default Dense layer
                if encoder:
                    act = self.activation['encoder']
                else:
                    act = self.activation['decoder'] if not layer_ind == len(dims)-1 else self.output_activation
                    print('*** ACTIVATION FOR LAYER ', layer_ind, act)
                print("ADDING DENSE : layer ind ", layer_ind)
                print('ind latent ', ind_latent, self.encoder_dims, layers_list)
                layer = layer_args.Layer(** {'type': 'Dense',
                                        'latent_dim': dims[layer_ind],
                                        'encoder': encoder,
                                        'layer_kwargs': {'activation': act,
                                            'kernel_initializer': self.initializer,
                                            'bias_initializer': self.initializer}})
        
            print('layer size ', layer.latent_dim, ' kw args: ', layer.layer_kwargs)
            # each function holds is a dictionary with 'stat' and 'act' (each of length k, with stats items [z_mean, z_var])
            functions = layer.make_function_list(index = layer_ind)

            # each k_layer is length k
            # each k_layer is called on the previous one (k_layer[1](k_layer[0]))
            # uses zip to call k = 0 index through layers 
            #       e.g. k_layer[2][0](k_layer[1][0](k_layer[0][0])))
            
            stat_k = functions['stat']
            act_k = functions['act']

            if stat_k:
                current = current_call[0]
                intermediate_stats = []
                for k in range(len(stat_k)):
                    # do all of these need to be same length?
                    stat_lyr = stat_k[k]
                    # needed to create list of z_mean, z_var to append to layers_list
                    for z in stat_lyr:
                        intermediate_stats.append(z(current))
                    layers_list[layer_ind]['stat'].append(intermediate_stats)
                current_call = layers_list[layer_ind]['stat']
                    # call act k layer on stat k [z_mean, z_var]
                    #for stat, act in zip(layers_list[layer_ind]['stat'], act_k):
                    #    print('stat ', stat, 'act ', act)
                    #    layers_list[layer_ind]['act'].append(act(stat))
            if len(act_k) < len(current_call):
                act_k = [a for idx in range(int(len(current_call)/len(act_k))) for a in act_k]


            for k in range(len(act_k)):
                act_lyr = act_k[k]
                if isinstance(current_call, list):
                    current = current_call[k] if len(current_call) > 1 else current_call[0]
                
                print("CURRENT (i.e. prev layer) ", current, current_call)
                a = act_lyr(current)
                layers_list[layer_ind]['act'].append(a)

                #for new, prev in zip(act_lyr, current_call):
                #layers_list[layer_ind]['act'].append(act_lyr(current))

            # for k_layer in functions:
            #     # each k_layer holds k functional layers (or lists, e.g. [z_mean,z_var]) 
            #     if len(inp) < len(k_layer):
            #         print("WARNING: Check this")
            #         inp = inp*len(k_layer)
            #     for k in range(len(k_layer)):
            #         lyr = k_layer[k]
            #         # Note, each lyr list will have length k 
            #         if isinstance(lyr, list):
            #             lyr_output = []
            #             # e.g. [z_mean(x), z_var(x)], which is called downstream
            #             for _lyr in lyr:
            #                 lyr_output.append(_lyr(inp[k]))
            #             layers_list[layer_ind].append(lyr_output)
            #         else:
            #             print(type(lyr), type(inp[k]))
            #             print('printed ')
            #             lyr_output = lyr(inp[k])
            #             layers_list[layer_ind].append(lyr_output)
             
            if encoder:
                print('encoder layer', layer_ind, ': ', layers_list[layer_ind])
            else:
                print('decoder layer', layer_ind, ': ', layers_list[layer_ind])
            #return self.encoder_layers
            
        return layers_list

    def get_generator(self):
        pass

    def get_encoder(self, encoder_args):
        pass

    def get_loss_history(self):
        pass

    def load_model(self, filename):
        try:
            model = model_from_json(open(filename).read())
        except:
            print('Error reading file: {0}. Cannot load previous model'.format(filename))
            exit()
        return model


    def load_weights(self, model, filename):
        try:
            #model()
            self = model.load_weights(filename)
        except:
            print('Error reading file: {0}. Cannot load previous weights'.format(filename))
            exit()


    def save_model(self, model, filename):
        json_string = model.to_json()
        open(filename, 'w').write(json_string)


    def save_weights(self, filename):
        model.save_weights(filename, overwrite=True)