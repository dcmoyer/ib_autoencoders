import matplotlib
matplotlib.use('Agg')
import numpy as np
import keras.backend as K
import tensorflow as tf
import importlib
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from keras import backend as K
from layers import Beta, MADE, MADE_network, IAF, Echo
from keras.layers import Input, Dense, merge, Lambda, Flatten #Concatenate, 
from keras.layers import Activation, BatchNormalization, Lambda, Reshape
from keras.callbacks import Callback, TensorBoard, LearningRateScheduler
import keras.models
import keras.layers
import keras.optimizers
import keras.initializers
#import utils
import layer_args
from loss_args import Loss
import dataset
import losses as l
import analysis
import pickle
import warnings
import callbacks

K.set_image_dim_ordering('tf')
RECON_LOSSES = ['bce', 'mse', 'binary_crossentropy', 'mean_square_error', 'mean_squared_error', 'iwae']

def load_model(filename):
        try:
            model = model_from_json(open(filename).read())
        except:
            print('Error reading file: {0}. Cannot load previous model'.format(filename))
            exit()
        return model

def load_weights(model, filename):
    try:
        #model()
        self = model.load_weights(filename+'_model')
    except:
        print('Error reading file: {0}. Cannot load previous weights'.format(filename))
        exit()


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
            pass # wrap transform
        if not callable(getattr(model, 'predict', None)):
            pass # wrap predict
        if not callable(getattr(model, 'generate', None)):
            pass # wrap generate

# TRANSFORM INTO Load from Config METHOD
# static methods creating, loading, saving, adding attributes


class NoiseModel(Model):
    def __init__(self, dataset, args_dict = {}, config = None, filename = 'model_stats', verbose = True):
        # dflt as dictionary (set by session)
        #if dflt is None:
        #   args = utils.load_from_config('configs/dflt.json')
        # All failed dictionary reads first check default config, then fall back to given value
        self.filename = filename
        self.verbose = verbose
        #'dataset': None, # specify + import dataset class
        # 'input_shape': None,
        self.args = {
            'per_label': None,
            'epochs': 100,
            'batch': 100,
            'optimizer': 'Adam',
            'initializer': 'glorot_uniform',
            'optimizer_params': {},
            'lr': 0.001,
            'lr_lagr': 0.01,
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
            'metrics': None,
            'constraints': None,
            'lagrangian_fit': False,
            'mismatch': None,
            'beta': 1.0,
            'anneal_schedule': None,
            'anneal_functions': None,
        }
        self.input_shape = None
        self.dataset = dataset
        if config is not None:
            if isinstance(config, dict):
                self.config = config
            else:
                try:
                    self.config = json.load(open(config))
                except:
                    self.config = json.load(open('configs/'+config))
            self.args.update(self.config)

        # read kwargs into dictionary
        self.args.update(args_dict) 
        for key in self.args.keys():
            setattr(self, key, self.args[key])

        self._parse_args()
        self._parse_layers_and_losses()
        #self._enc_ind_args = []

        # initialize dictionary (with keys = layer index) of dict of called layers (keys = 'stat', 'act')
        self.encoder_layers = [] #defaultdict(dict)
        self.decoder_layers = [] #defaultdict(dict)
        self.density_estimators = []#defaultdict(list)
        self.density_callback = False

    def _parse_args(self):
        
        if self.dataset == 'mnist':
            self.dataset = dataset.MNIST(binary = False)
        elif self.dataset == 'binary_mnist':
            self.dataset = dataset.MNIST(binary = True)
        elif self.dataset == 'omniglot':
            self.dataset = dataset.Omniglot()
        elif self.dataset == 'celeb_a':
            pass
        elif self.dataset == 'dsprites':
            self.dataset = dataset.DSprites()
        
        if self.per_label is not None:
            self.dataset.shrink_supervised(self.per_label)

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
                self.lr_callback = isinstance(self.lr, str)
                self.lr = getattr(mod, self.lr)
            except:
                try:
                    mod = importlib.import_module(str('custom_functions.lr_sched'))
                    # LR Callback will be True /// self.lr = function of epochs -> lr
                    self.lr_callback = isinstance(self.lr, str)
                    self.lr = getattr(mod, self.lr)
                except:
                    #self.lr = dflt.get('lr', .001)
                    print()
                    warnings.warn("Cannot find LR Schedule function.  Proceeding with default, constant learning rate.")    
                    print()
        print()
        print("**** LR CALLBACK *** ", self.lr_callback, self.lr)
        print()
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

        if self.decoder_dims is None or self.decoder_dims[-1] != 1: # check if has decoder_dims arg
            self.decoder_dims = list(reversed(self.encoder_dims[:-1]))
            self.decoder_dims.append(self.dataset.dim)
        else:
            #if self.decoder_dims[-1] != self.dataset.dim:
            #    self.decoder_dims.append(self.dataset.dim)
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
        self.anneal = (self.anneal_schedule is not None or self.anneal_functions is not None)
        if self.anneal_functions is not None:# and not isinstance(self.anneal_functions, dict):
            if isinstance(self.anneal_functions, str):
                self.anneal_functions = {0: self.anneal_functions}
                warn("Anneal Function not a dictionary.  Default is to apply to LOSS index 0")
            elif isinstance(self.anneal_functions, list):
                anneal_copy = self.anneal_functions.copy()
                self.anneal_functions = {}
                for l in range(len(anneal_copy)):
                    self.anneal_functions[l] = anneal_copy[l]

            anneal_copy = self.anneal_functions.copy()
            for k in anneal_copy.keys():
                print("Key ", k)
                try:
                    mod = importlib.import_module('.'.join(self.anneal_functions[k].split('.')[:-1]))
                except:
                    mod = importlib.import_module('custom_functions.anneal_fn')
                # LR Callback will be True /// self.lr = function of epochs -> lr
                self.anneal_functions[int(k)] = getattr(mod, self.anneal_functions[k].split('.')[-1])
                del self.anneal_functions[k]
            print()
            print('*** ANNEAL FUNCTION ***')
            print(self.anneal_functions)

        self.lagrangian_fit = (self.constraints or self.constraints is not None) or (self.mismatch or self.mismatch is not None) #or not (not self.mismatch))
        
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
        print(self.layers[0])
        if self.metrics is not None:
            for metric in self.metrics:
                if metric['weight'] != 0:
                    warn("Metric ", metric['type'], " weight is non-zero. Setting to 0.  Enter as loss to add to objective")
                    metric['weight'] = 0
                self.losses.append(metric)

        # loop through to record which layers have losses attached
        if self.losses is not None and isinstance(self.losses, list):
            pass # TESTING IF WE CAN REMOVE THIS
            # for i in range(len(self.losses)):
            #     lossargs = self.losses[i]
            #     if lossargs.get('encoder', True) and (lossargs.get('type') not in RECON_LOSSES and lossargs.get('add_loss') not in RECON_LOSSES): # loss defaults to encoder unless in recon
            #         if 'encoder' not in lossargs:
            #             warn("Loss entry ", i, ": ",  lossargs.get('type', '') ," defaulting to encoder")

            #         self._enc_loss_ind.append(len(self.encoder_dims)-1 
            #                                     if lossargs.get('layer', -1) == -1
            #                                     else lossargs['layer'])
            #     else:
            #         self._dec_loss_ind.append(len(self.decoder_dims)-1 
            #                                     if lossargs.get('layer', -1) == -1
            #                                     else lossargs['layer'])
            #         if lossargs.get('layer', -1) != -1:
            #             print("WARNING: loss on intermediate decoder layers doesn't seem to make sense.  If you want a Corex layer to reconstruct a decoder layer, specify via layer arguments and mapping btwn indices of layer_args list and latent_dims list")
        else:
            self.losses = []
            print("WARNING: Losses not specified as list of dictionaries?  DEFAULTING to Recon argument (+ any default noise layers specified) ")

        # loop through to record which layers have special layer arguments 
        # (noise regularization or non-dense layer type)
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

    # TO DO : MAKE FIT AUTOMATICALLY READ DATASET
    def fit(self, x_train, y_train = None, x_val = None, y_val = None, verbose = None):
        if verbose is not None:
            self.verbose = verbose

        print("INPUT SHAPE ", self.input_shape)
        if self.input_shape is None:
            self.input_shape = (self.dataset.dims[0], self.dataset.dims[1], 1) if 'Conv' in self.layers[0]['type'] else (self.dataset.dim,)
            print(self.dataset)
            print(self.dataset.dim, self.dataset.dims)
        self.input_tensor = Input(shape = (self.dataset.dim,)) 
        print("INPUT SHAPE ", self.input_shape)
        print("INPUT TENSOR ", self.input_tensor)
        if self.input_shape is not None:
            self.input = Reshape(self.input_shape)(self.input_tensor)
        else:
            self.input = self.input_tensor    
        self.recon_true = self.input_tensor # Lambda(lambda y : y, name = 'x_true')(x)
        #print('INPUT SHAPE ', self.input_shape, ' x shape ', self.input_tensor, self.input_shape.insert(-1, 0) if self.input_shape is not None else '')
        #print("***********************         ENCODER          *****************************")
        self.encoder_layers = self._build_architecture([self.input], encoder = True)
        #print("***********************         DECODER          *****************************")
        self.decoder_layers = self._build_architecture(self.encoder_layers[len(self.encoder_layers)-1]['act'], encoder = False)
        self.model_outputs, self.model_losses, self.model_loss_weights = self._make_losses()
        #self.metric_outputs, self.metric_losses, self.metric_weights = self._make_losses(metrics = True)
        try:
            self.dc_model = keras.models.Model(inputs = self.input_tensor, outputs = self.dc_outputs)
        except:
            pass

        
        #print(self.model_outputs)
        #print([type(o) for o in self.model_outputs])
        self.model = keras.models.Model(inputs = self.input_tensor, outputs = self.model_outputs)
        
        print(self.model.summary())

        print(self.model_loss_weights)

        for i in self.model.layers[1:]:
            try:
                print(i.name, i.activation)
            except:
                pass
        #print('outputs ', self.model_outputs)
        #print('losses ', self.model_losses)
        
        #print("optimizer ", self.optimizer)
        #print("ENTROPY OF DATA ", np.mean(np.sum(np.multiply(x_train, np.log(x_train+10**-7)), axis = -1)))
        
        self.model.compile(optimizer = self.optimizer, loss = self.model_losses, loss_weights = self.model_loss_weights) # metrics?
        print(self.model.metrics_names)
        callbacks = self._make_callbacks()
        print(callbacks)


        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()
         
        #print(self.lagrangian_fit)
        if not self.lagrangian_fit:
            #self.model.compile(optimizer = self.optimizer, loss = self.model_losses, loss_weights = self.model_loss_weights) # metrics?
            hist = self.model.fit(x_train, ([x_train] if y_train is None else [y_train])*len(self.model_outputs), 
                               epochs = self.epochs, batch_size = self.batch, callbacks = callbacks, verbose = self.verbose)
            self.hist= hist.history
        else:
            self.lagrangian_optimization(x_train, y_train, x_val, y_val)
            #self.model.compile(optimizer = self.optimizer, loss = self.model_losses, loss_weights = self.model_loss_weights) # metrics?
        
        self.test_eval()
        self.pickle_dump()
        self.save_model(self.filename)
        # how to get activation layers?
        examples = x_train[:self.batch]
        z = self._encoder(x = examples)
        try:
            x_pred = self._decoder()            
        except:
            x_pred = self._decoder(K.expand_dims(K.expand_dims(z,2), 3))

        means = np.mean(z, axis = 0)
        sigs = np.sqrt(np.var(z, axis = 0))

        print("FILENAME ", self.filename)
        analysis.plot_traversals(examples, 
                        self._encoder(), 
                        self._decoder(),
                        z_act = z,
                        imgs = 3,
                        means = means,
                        sigmas = sigs,
                        prefix = self.filename)
        print('enc latent ind' , len(self._enc_latent_ind)-1)
        print(' with total layers= ', len(self.layers))
        print('layer ', self.layers[len(self._enc_latent_ind)-1])
        analysis.sample_echo_reconstructions(self.model, self.dataset, echo_batch = self.batch, echo_dmax = self.layers[len(self._enc_latent_ind)-1]['layer_kwargs']['d_max'],
            prefix = self.filename)

        if self.encoder_dims[-1] == 2:
            analysis.manifold(z, x_pred, per_dim = 50, dims = self.dataset.dims, location = 'results/'+self.filename)
        
            if y_train is None:
                y_train = self.dataset.y_train

            analysis.two_d_labeled(x_train, y_train, self._encoder(), batch = self.batch, prefix = self.filename)

        #tf_mod(kz, x_out, sess), top = latent_dims[-1], prefix = os.path.join(os.path.dirname(os.path.realpath(__file__)), log_path, '_'), z_act = z_acts, means= means, sigmas = sigs, imgs = p)


    def test_eval(self, x_test = None, y_test = None):
        self.test_results = {}
        if x_test is None:
            try:
                x_test = self.dataset.x_test
            except:
                print("X Test: ", x_test.shape)
            if x_test is None:
                raise ValueError('Please feed test data to test_eval method or Dataset object')
        #preds = self.model.predict(x_test, batch_size = self.batch)
        
        #preds = self.sess.run(self.model.outputs,
        #            feed_dict={self.input_tensor: x_test})
        
        try:
            find_recon = [i for i in range(len(self.model.metrics_names)) if 'recon' in self.model.metrics_names[i]]
            print("Output TENSOR SHAPE : ", K.int_shape(self.model.outputs[find_recon[0]]))
        except:
            find_recon = [-1]

        if self.lagrangian_fit:
            inps = None
            
            if K.int_shape(self.model.outputs[find_recon[0]])[0] is None:
                for offset in range(0, (int(x_test.shape[0] / self.batch) * self.batch), self.batch):  # inner
                    batch_data = x_test[offset:(offset + self.batch)]
                    new_inps = self.sess.run(self.loss_inputs,
                        feed_dict={self.input_tensor: batch_data})
                    #print("new inps ", type(new_inps), " len ", len(new_inps), ": ", [type(new_inps[i]) for i in range(len(new_inps))])
                    inps = [[tf.concat([inps[i][j], new_inps[i][j]], axis = 0) for j in range(len(inps[i]))] for i in range(len(inps))] if inps is not None else new_inps
                    #inps = [inps[i] + new_inps[i] for i in range(len(new_inps))] if inps is not None else new_inps
                    #("inps shape ", inps[0][0].shape, inps[1][0].shape, inps[2][0].shape)#, inps[3][0].shape)
                # average each loss over batches
                #inps = [[inps[i][j]/(int(x_test.shape[0]/self.batch)) for j in range(len(inps[i]))] for i in range(len(inps))]
            else:
                inps = self.sess.run(self.loss_inputs,
                        feed_dict={self.input_tensor: x_test})

            for i in range(len(inps)):
                lv = l.loss_val(self.loss_functions[i]([K.variable(tensor) for tensor in inps[i]])).eval(session=K.get_session())
                print("Test loss ", self.model.metrics_names[i+1], " : ", lv)
                self.test_results[self.model.metrics_names[i+1]] = lv
        # for i in range(len(self.loss_functions)):
        #     print("loss function ", self.loss_functions[i])
        #     loss_value = self.loss_functions[i](self.loss_inputs[i])
        #     loss_value = l.loss_val(loss_value)
        #     print()
        #     print("Loss ", self.model.metrics_names[i+1], " : ", self.sess.run(loss_value, feed_dict = {self.input_tensor: x_test}))
        

        #plt.imsave(arr = bce_pred[-1, :].reshape((28,28)), fname = str('test.png'))
        #print("BCE test ", l.loss_val(l.binary_crossentropy([K.variable(x_test), K.variable(bce_pred)])).eval(session=K.get_session()))
        else:
            # OLD APPROACH (wrong values for lagrangian)
            loss_list = self.model.evaluate(x_test, [x_test]*len(self.model.outputs), batch_size = self.batch)
            
            for i in range(len(loss_list)):
                self.test_results[self.model.metrics_names[i]] = loss_list[i]
                print("Test loss ", self.model.metrics_names[i], " : ", loss_list[i])

    def _encoder(self, x = None):
        for i in self.model.layers:
            print(i.name)
            if 'z_act' in i.name or 'echo' in i.name:
                final_latent = i.name
                break
        print("FINAL LATENT ", final_latent)
        get_z = K.function([self.model.layers[0].get_input_at(0)], [
                        self.model.get_layer(final_latent).get_output_at(0)])
        if x is not None:
            print("GET Z LIST ", len(get_z([x])), get_z([x])[0].shape) 
        return get_z if x is None else get_z([x])[0]

    def _encoder_stats(self, x = None):
        # CHANGES FOR ECHO
        for i in self.model.layers:
            if 'z_mean' in i.name:
                mean_latent = i.name
            if 'z_var' in i.name:
                var_latent = i.name

        get_z_mean = K.function([self.model.layers[0].get_input_at(0)], [
                        self.model.get_layer(mean_latent).get_output_at(0)])
        try:
            get_z_var = K.function([self.model.layers[0].get_input_at(0)], [
                        self.model.get_layer(var_latent).get_output_at(0)])
        except:
            get_z_var = None
        return get_z_mean, get_z_var if x is None else get_z_mean([x])[0], get_z_var([x])[0]
        #return get_z if x is None else get_z([x])[0]    

    def _decoder(self, z = None):
        for i in self.model.layers:
            if 'z_act' in i.name or 'echo' in i.name:
                final_latent = i.name
                break

        z_inp = Input(shape = (self.encoder_dims[-1],))
        z_out = z_inp
        call_ = False
        for layr in self.model.layers:
            # only call decoder layer_list
            if call_ and not isinstance(layr, keras.layers.core.Lambda) and not isinstance(layr, MADE) and not isinstance(layr, MADE_network):#and not ('vae' in layr.name or 'noise' in layr.name or 'info_dropout' in layr.name):
                z_out = layr(z_out)
            if layr.name == final_latent:
                call_ = True
            # doesn't work with new naming convention
            #if layr.name == 'decoder' or layr.name == 'ci_decoder':
            #    call_ = False
        generator = keras.models.Model(input = [z_inp], output = [z_out])
        return generator if z is None else generator.predict(z) 

    def _make_losses(self, metrics = False):
        if metrics:
            loss_list = self.metrics
            self.metric_outputs = [] 
            self.metric_losses = [] 
            self.metric_weights = []
        else:
            loss_list = self.losses
            self.model_outputs = []
            self.model_losses = []
            self.model_loss_weights = []
        
        
        if loss_list is not None:
            #for ls in ['dec', 'enc']:
            for i in range(len(loss_list)):
                callback = False
                print('loss type ', loss_list[i])
                #print(vars(loss_list[i]))
                loss = Loss(**loss_list[i]) if isinstance(loss_list[i], dict) else loss_list[i]

                #loss_list[i] = loss
                print("Loss constraint value: ", loss.constraint)
                #if loss.constraint is not None:
                #    self.lagrangian_fit = True

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
                try:
                    self.loss_functions.append(loss.make_function())
                except:
                    self.loss_functions = []
                    self.loss_functions.append(loss.make_function())

                print('loss func *********', loss.type, loss.layer)
                inputs_layer, inputs_output = loss.describe_inputs()
                outputs = []
                print('describe inputs', inputs_layer, inputs_output)
                
                for j in range(len(inputs_layer)): # 'stat' or 'act'
                    # enc / dec already done
                    layers = self.encoder_layers if enc else self.decoder_layers
                    layers = layers if not 'density' in loss.type else self.density_estimators
                    if loss.callback:
                        self.density_callback = True 
                        callback = True
                        try:
                            a = self.dc_outputs[0]
                        except:
                            self.dc_outputs = []
                            self.dc_losses = [] 
                            self.dc_weights = []
                            self.dc_names = {}

                    #if 'density' in loss.type:
                        #print('density ', list(self.density_estimators.keys()))
                    lyr = loss.layer 
                     
                    #  'stat' or 'act' for enc/dec layer # lyr
                    if 'act' in inputs_layer[j] or 'addl' in inputs_layer[j]:
                        #if not loss.encoder and lyr in [-1, len(self.decoder_dims)-1]:
                        #    layers[lyr]['act'].insert(0, self.recon_true)
                        #if isinstance(layers[lyr][inputs_layer[j]], list):
                        #    outputs.extend([l for l in layers[lyr][inputs_layer[j]]])
                        #else:
                        outputs.extend(layers[lyr][inputs_layer[j]])
                        print("*** adding (act/addl) ", layers[lyr][inputs_layer[j]])
                        print(outputs)
                    elif 'stat' in inputs_layer[j]:
                        try:
                            outputs.extend(layers[lyr][inputs_layer[j]][0])
                            print('adding stat from ', layers[lyr], 'choosing lyr_arg', inputs_layer[j])
                        except:
                            print('adding stat from ', layers[-1], 'choosing lyr_arg', inputs_layer[j])
                            outputs.extend(layers[-1][inputs_layer[j]])

                for j in range(len(inputs_output)):
                    layers = self.decoder_layers
                    lyr = loss.output
                    if 'act' in inputs_output[j]:# == 'act':
                        # all output activations get either recon_true or encoder activation (for corex)
                        if (lyr == -1): #[-1, len(self.decoder_dims)-1]):
                            recon_true = self.recon_true #if len(K.int_shape(self.recon_true)) == 2 else Flatten()(self.recon_true) 
                            #K.reshape(self.recon_true, [-1, np.prod([i for i in self.recon_true.get_shape().as_list()[1:] if i is not None])])
                        else:
                            if len(self.encoder_layers[lyr]['act']) == 1:
                                recon_true = layers[lyr]['act'][0]
                                print('rECON TrUE ', recon_true) 
                            else:
                                raise NotImplementedError("Cannot handle > 1 activation for intermediate layer reconstruction")
                        
                        layers[lyr][inputs_output[j]].insert(0, recon_true)
                        #print('act output', layers[lyr][inputs_output[j]])
                        outputs.extend(layers[lyr][inputs_output[j]])
                    elif 'stat' in inputs_output[j]:
                        #print('stat output', layers[lyr][inputs_output[j]])
                        outputs.extend(layers[lyr][inputs_output[j]][0])
                    # not handling 'addl' tensors

                #print('outputs for layer ', outputs, [K.int_shape(o) for o in outputs])
                try:
                    for j in range(len(outputs)):
                        #outputs[j] = Flatten()(outputs[j]) #
                        #try:
                            #outputs[j] = Reshape([-1, *outputs[j]._keras_shape[1:]])(outputs[j]) if len(K.int_shape(outputs[j])) > 2 else outputs[j]
                        #except:
                        outputs[j] = Flatten()(outputs[j]) if len(K.int_shape(outputs[j])) > 2 else outputs[j]
                        #outputs[j] = tf.reshape(outputs[j], [-1, K.int_shape(outputs[j])[-1]]) if len(K.int_shape(outputs[j])) > 2 else outputs[j]
                except Exception as e:
                    if not 'density' in loss.type:
                        print("***Exception***", e)
                    for j in range(len(outputs)):
                        print(outputs[j])
                        for k in range(len(outputs[j])):
                            outputs[j][k] = Flatten()(outputs[j][k]) if len(K.int_shape(outputs[j][k])) > 2 else outputs[j][k]
                
                #[-1, np.prod([i for i in o.get_shape().as_list()[1:] if i is not None])]) for o in outputs]
                #o.get_shape().as_list()[1:])]) for o in outputs]
                #outputs = [tf.contrib.layers.flatten(output) for output in outputs]
                if metrics:
                    self.metric_outputs.append(self.loss_functions[-1](outputs))
                    self.metric_losses.append(l.dim_sum) 
                    self.metric_weights.append(loss.get_loss_weight())
                elif callback:
                    print("ENTERING CALLBACK ", self.loss_functions[-1])
                    print('outputs', outputs)
                    self.dc_outputs.append(self.loss_functions[-1](outputs[0]))
                    self.dc_losses.append(l.mean_one) 
                    self.dc_weights.append(loss.get_loss_weight())
                    self.dc_names[loss.name] = len(self.dc_losses)
                else:
                    try:
                        self.loss_inputs.append(outputs)
                    except:
                        self.loss_inputs = []
                        self.loss_inputs.append(outputs)
                    try:
                        self.model_outputs.append(self.loss_functions[-1](outputs))
                    except:
                        print("outputs ", outputs)
                        self.model_outputs.append(self.loss_functions[-1](outputs[0]))
                    self.model_losses.append(l.dim_sum)
                    self.model_loss_weights.append(loss.get_loss_weight())
                print("OUTPUTS ", self.model_outputs)
                print("Losses: ", self.model_losses)

            def reg_losses(tensors):
                print('tensors ', [t.name for t in tensors])
                return [K.sum(t, keepdims=True) for t in tensors if 'reg' in t.name]
            
            #if len(reg_losses(self.model_outputs))>1:
            #    self.model_outputs.append(Lambda(tf.add_n, name = 'sum_regs')(Lambda(reg_losses, name = 'add_regs')(self.model_outputs)))
            #    self.model_losses.append(l.dim_sum)
            #    self.model_loss_weights.append(0)

            # losses are all Lambda layers on tensors to facilitate optimization in tensorflow
                # return dimension-wise loss and then sum in dummy loss function?

            # allow returning dimension-wise loss for plot_traversals, e.g.  
            # mean and sum over batch / dimensions should be done here

        if metrics:
            return self.metric_outputs, self.metric_losses, self.metric_weights
        else:
            return self.model_outputs, self.model_losses, self.model_loss_weights

    def _make_callbacks(self):
        my_callbacks = []
        if self.lr_callback:
            my_callbacks.append(LearningRateScheduler(self.lr))
        print()
        #print("MAKING CALLBACKS ", self.anneal_functions)
        if self.anneal_functions:
            
            annealed_losses = []
            for lw in range(len(self.model_loss_weights)):
                if (lw) in [int(i) for i in list(self.anneal_functions.keys())]:
                    loss_weight_tensor = tf.Variable(self.anneal_functions[lw](1),  trainable = False, name ='annealed_weight_'+str(lw))
                    #loss_weight_layer.trainable = False
                    self.model_loss_weights[lw] = loss_weight_tensor
                    annealed_losses.append(loss_weight_tensor)
            
            my_callbacks.append(callbacks.BetaCallback(functions= self.anneal_functions, layers = annealed_losses))
        
        if self.density_callback:
            #for i in range(len(self.dc_losses)):
            dc = callbacks.DensityTrain(self.dc_model, self.dc_names, self.dc_outputs, self.dc_losses, self.dc_weights)
            my_callbacks.append(dc)

            #dc2 = callbacks.DensityEpoch(self.dc_model, self.dc_names, self.dc_outputs, self.dc_losses, self.dc_weights, batch = self.batch)
            #my_callbacks.append(dc2)

            layer_name = self.encoder_layers[-1]['act'][0].name
            #print("LAYER NAME ", layer_name, ' act ', self.encoder_layers[-1]['act'])
            #print("model names ", [i.name for i in self.model.layers])
            fetches = [tf.assign(dc.x, self.model.targets[0], validate_shape=False),
                tf.assign(dc.z, self.model.get_layer(layer_name.split("/")[0]).get_output_at(0)[0], validate_shape=False)]
                
            self.model._function_kwargs = {'fetches': fetches}

        return my_callbacks
        print()

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
            offset = 0
            #ind_args = self._enc_ind_args
        else:
            #self.decoder_layers = defaultdict(list)
            layers_list = self.decoder_layers
            dims = self.decoder_dims
            ind_latent = self._dec_latent_ind
            offset = len(self.encoder_layers)
            #ind_args = self._dec_ind_args
        print()
        print(layers_list)
        print()

        for layer_ind in range(len(dims)):
            #if layer_ind == len(dims):
                # if prev layer is convolutional
            density = None
            if encoder:
                #self.encoder_layers = defaultdict(list)
                layers_list = self.encoder_layers
                dims = self.encoder_dims
                ind_latent = self._enc_latent_ind
                offset = 0
                #ind_args = self._enc_ind_args
            else:
                #self.decoder_layers = defaultdict(list)
                layers_list = self.decoder_layers
                dims = self.decoder_dims
                ind_latent = self._dec_latent_ind
                offset = len(self.encoder_layers)
                #ind_args = self._dec_ind_args
            print("***** PROCESSING LAYER ***** ", "encoder " if encoder else "decoder ", layer_ind)
            print("layers list ", layers_list, layer_ind)

            if layer_ind == 0:
                if input_list is not None:
                    current_call = input_list
                else:
                    raise ValueError("Feed input tensor to _build_architecture")
            else:
                # list of inputs ([] to start)
                print("layers list ", layers_list, layer_ind-1)
                try:
                    current_call = layers_list[layer_ind-1]['act']
                except:
                    current_call = self.decoder_layers[layer_ind-1]['act']

            if layer_ind in ind_latent:
                # retrieve layer_arguments index for given encoding layer
                arg_ind = ind_latent.index(layer_ind) + offset
                if self.layers[arg_ind].get('latent_dim', None) is None:
                    self.layers[arg_ind]['latent_dim'] = dims[layer_ind]
                print("ADDING DIM ", dims[layer_ind], layer_ind, ind_latent)
                print(self.layers[arg_ind])

                if "echo" in self.layers[arg_ind]['type'] or self.layers[arg_ind]['type'] in ['bir', 'constant_additive']:
                    self.layers[arg_ind]['layer_kwargs']['batch'] = self.batch 
                    
                try:
                    if self.layers[arg_ind]['density_estimator']['type'] in ['maf', 'masked_arf']: 
                        self.layers[arg_ind]['density_estimator']['latent_dim'] = self.layers[arg_ind]['latent_dim']
                        density = layer_args.Layer(**self.layers[arg_ind]['density_estimator']) 
                except:
                    pass
                #:if self.layers[arg_ind]['type'] in ['maf', 'masked_arf']:
                    #self.density_estimators
                    #layers_list = self.density_estimators
                    #or self.layers[arg_ind]['type'] in ['bir', 'constant_additive']:


                layer = layer_args.Layer(** self.layers[arg_ind])
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

            for mode in range(2):
                if mode == 1:
                    print("*** DENSITIES *** ", density)
                    if density is not None:
                        functions = density.make_function_list(index = layer_ind)
                        print("functions ", functions)
                        layers_list = self.density_estimators
                        layers_list.append(defaultdict(list))
                    else:
                        continue
                        #self.density_estimators.append(defaultdict(list))
                        #continue
            # add for recording 'stat', 'act', etc.  moved to account for self.density_estimators (MADE changes layers_list)
                else:   
                    layers_list = self.encoder_layers if encoder else self.decoder_layers
                    layers_list.append(defaultdict(list))
                    #print('layer size ', layer.latent_dim, ' type ', layer.type,  ' kw args: ', layer.layer_kwargs)
                    # each function holds is a dictionary with 'stat' and 'act' (each of length k, with stats items [z_mean, z_var])
                    functions = layer.make_function_list(index = layer_ind)

                # each k_layer is length k
                # each k_layer is called on the previous one (k_layer[1](k_layer[0]))
                # uses zip to call k = 0 index through layers 
                #       e.g. k_layer[2][0](k_layer[1][0](k_layer[0][0])))
                
                stat_k = functions['stat']
                act_k = functions['act']
                addl_k = functions['addl']
                try:
                    call_on_addl = functions['call_on_addl']
                except:
                    call_on_addl = None

                print("layer functions: stats ", functions['stat'], ' act ', functions['act'], ' addl ', functions['addl'])
                if stat_k:
                    current = current_call[0]
                    intermediate_stats = []
                    for k in range(len(stat_k)):
                        # do all of these need to be same length?
                        stat_lyr = stat_k[k]
                        # needed to create list of z_mean, z_var to append to layers_list
                        for z in stat_lyr:
                            intermediate_stats.append(z(current))
                        try:
                            layers_list[layer_ind]['stat'].append(intermediate_stats)
                            current_call = layers_list[layer_ind]['stat']
                        except: 
                            layers_list[-1]['stat'].append(intermediate_stats)
                            current_call = layers_list[-1]['stat']
                            
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
                    act = act_lyr(current) # if not echo_flag else act_lyr(current)[0]
                    try:
                        layers_list[layer_ind]['act'].append(act)
                        current_call = layers_list[layer_ind]['act']
                    except:
                        layers_list[-1]['act'].append(act)
                        print("ACTIVATION TYPE ", act, type(act))
                        if isinstance(act, IAF):
                            current_call = layers_list[-1]['act']
                        # ADDED FOR MADE, which doesn't call on activation layer?
                

                for k in range(len(addl_k)):
                    print()
                    print("echo capacity layer:")
                    print("CURRENT (i.e. prev layer) ", current, current_call)
                    addl_lyr = addl_k[k]
                    print("addl_lyr")
                    if isinstance(current_call, list):
                        print("current is a list")
                        current = current_call[k] if len(current_call) > 1 else current_call[0]
                    else:
                        current = [current]
                    print("calling on input: ", current)
                    a = addl_lyr(current)
                    print("done calling")
                    try:
                        layers_list[layer_ind]['addl'].append(a)
                        #current_call = layers_list[layer_ind]['act']
                    except:
                        layers_list[-1]['addl'].append(a)
                        #current_call = layers_list[-1]['act']
                    print("appended")
                if call_on_addl is not None:
                    for k in range(len(call_on_addl)):
                        print("CURRENT (i.e. prev layer) ", current, current_call)
                        if isinstance(current_call, list):
                            print("current is a list")
                            current = current_call[k] if len(current_call) > 1 else current_call[0]
                        else:
                            current = [current]
                        
                        a = addl_lyr(current)
                        layers_list[-1]['act'].append(a)
                    #print("ADDL for last LAYER ", layers_list[layer_ind]['addl'])
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
                layers_list = self.encoder_layers
                #print('encoder layer', layer_ind, ': ', layers_list[layer_ind])
            else:
                layers_list = self.decoder_layers
                #print('decoder layer', layer_ind, ': ', layers_list[layer_ind])
            #return self.encoder_layers
            
        return layers_list


    def lagrangian_optimization(self, x_train, y_train = None, x_val = None, y_val = None, 
                   min_lagr = .001, max_lagr = 100.0):

        lagrangians = []
        lagr_vars = []
        if self.mismatch:
            hidden_ind = [j for j in range(len(self.layers)) if self.config['layers'][j]['encoder']][-1]
            conf = self.config.copy()
            conf['layers'][hidden_ind]= {'type':'additive', 'layer':-1, 'k':1, 'encoder': True} #self.mismatch['type']
            conf['losses'] = []
            conf['constraints'] = []
            self.build_mismatch_model(conf)
            # note: don't train decoder, but share weights...
            print("mismatch loss (outputs) ", self.mm.model.outputs)
            mismatch_loss = self.mm.loss_weights*l.loss_val(self.mm.model.outputs[-1])
            # CAREFUL WITH TRAINABLE VARIABLES (particularly since above trainer/lagr works off of all)
            train_mm = tf.train.AdamOptimizer(self.lr if not self.lr_callback else self.lr(0)).minimize(mismatch_loss, var_list = self.mm.model.trainable_weights)
            
            mismatch_init = 1.0
            sign = 1
            multiplier = tf.get_variable("lagr_mm", initializer = mismatch_init, dtype = tf.float32)#, name = ) 
            lagr_vars.append(multiplier)
            #lagrangian_loss =  multiplier*(K.abs(l.loss_val(self.model.outputs[-1] - self.mm.model.outputs[-1])))
            print(self.model.outputs, ' metrics ', self.model.metrics_names)
            lagrangian_loss =  multiplier*(K.abs(l.loss_val(self.model.outputs[-1])-l.loss_val(self.mm.model.outputs[-1])))
            lagr_loss = -1#index


        else: # constraint
            for i in range(len(self.constraints)):
                constraint = self.constraints[i]
                init = self.model_loss_weights[constraint['loss']]*1.0
                try:
                    sign = -1 if 'geq' in constraint['relation'] or 'greater' in constraint['relation'] else 1
                except:
                    sign = 1

                multiplier = tf.get_variable("lagr_"+str(i), initializer = init*sign, dtype = tf.float32)#, name = ) 
                loss_tensor = self.model.outputs[constraint['loss']] #_losses[constraint['loss']]
                
                #if not constraint.get('value', False) or constraint['value'] == 'prior':
                #    constraint['value'] = l.gaussian_prior_kl(self.encoder_layers[-1]['stat'])

                lagrangians.append(multiplier*(l.loss_val(loss_tensor) - tf.constant(constraint['value']*1.0))) #= multiplier*(l.dimsum(loss_tensor) - constraint['value'])
                lagr_vars.append(multiplier)
                lagr_loss = constraint['loss']

            lagrangian_loss = tf.add_n(lagrangians) 

        total_loss = lagrangian_loss + tf.add_n([self.model_loss_weights[i]*l.loss_val(self.model.outputs[i]) for i in range(len(self.model.outputs)) if i != lagr_loss]) 
        
        # all other parts of objective... 
        other_vars = [v for v in self.model.trainable_weights if "lagr" not in v.name]
        #other_vars = [v for v in tf.trainable_variables() if "lagr" not in v.name]

        print("other vars ", other_vars)
        print("LAGR VARS ", lagr_vars)

        
        learning_rate = tf.placeholder(tf.float32)
        trainer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, var_list=other_vars)
        lagrangian_trainer = tf.train.GradientDescentOptimizer(self.lr_lagr).minimize(-lagrangian_loss, var_list=lagr_vars)
        
        # DOESN'T WORK WITH > 1 CONSTRAINT
        lagr_clip = tf.assign(lagr_vars[0], sign*tf.minimum(tf.maximum(lagr_vars[0], min_lagr), max_lagr))
        #
        #tf.group( )
         #   *[tf.assign(lagr, tf.minimum(tf.maximum(lagr, min_lagr), max_lagr)) for lagr in lagr_vars])
            #[tf.assign(self.l1, tf.minimum(tf.maximum(self.l1, 0.001), 100.0)),
            #tf.assign(self.l2, tf.minimum(tf.maximum(self.l2, 0.001), 100.0))
        #)
        if self.anneal_functions:
            anneal_loss_weights = self.anneal_loss_weights()
            update_lw = tf.assign(anneal_loss_weights[l], self.anneal_functions[l](epoch))


        n_samples = x_train.shape[0]
        self.sess = tf.Session()
        self.hist = defaultdict(list)
        with self.sess.as_default():
            tf.global_variables_initializer().run()
            for i in range(self.epochs):  # Outer training loop
                epoch_avg = defaultdict(list)
                total_avg = []
                lagr_avg = []
                lm_avg = []
                perm = np.random.permutation(n_samples)  # random permutation of data for each epoch
                
                if self.anneal_functions:
                    self.sess.run(update_lw)


                for offset in range(0, (int(n_samples / self.batch) * self.batch), self.batch):  # inner
                    
                    batch_data = x_train[perm[offset:(offset + self.batch)]]
                    if self.mismatch:
                        _, mm_loss, _, _, tl= self.sess.run([train_mm, mismatch_loss, trainer, lagrangian_trainer, total_loss],
                                           feed_dict={self.input_tensor: batch_data, learning_rate: self.lr if not self.lr_callback else self.lr(i)})
                        epoch_avg['mismatch_recon'].append(np.mean(mm_loss))
                        epoch_avg['total_loss'].append(np.mean(tl))
                    else:
                        result = self.sess.run([trainer, lagrangian_trainer, total_loss],
                                           feed_dict={self.input_tensor: batch_data, learning_rate: self.lr if not self.lr_callback else self.lr(i)})
                        epoch_avg['total_loss'].append(np.mean(result[-1]))
                    self.sess.run(lagr_clip)
                    #for j in range(len(lagr_vars)):
                    #    tf.clip_by_value(lagr_vars[j], min_lagr, max_lagr)
                    # every epoch record loss
                    for loss_layer in self.model.outputs:
                        # SLOW?  fix recording
                        batch_loss = self.sess.run(loss_layer, feed_dict={self.input_tensor: batch_data})
                        epoch_avg[loss_layer.name].append(np.mean(np.sum(batch_loss, axis = -1), axis = 0))
                    
                    # batch_loss = self.sess.run([ll for ll in self.model.outputs], feed_dict={self.input_tensor: batch_data})
                    # for i in range(len(self.model.outputs)):
                    #     # SLOW?  fix recording
                    #     epoch_avg[self.model.outputs[i].name].append(np.mean(np.sum(batch_loss[i], axis = -1), axis = 0))


                    #print(self.sess.run(tf.group([i for i in lagr_vars])))#, feed_dict = {self.input_tensor:batch_data}))
                    lm_avg.append(self.sess.run(multiplier, feed_dict = {self.input_tensor:batch_data}))
                    #total_avg.append(self.sess.run(total_loss, feed_dict = {self.input_tensor: batch_data}))
                    #lagr_avg.append(self.sess.run(lagrangian_loss, feed_dict = {self.input_tensor:batch_data}))
                    #others = self.sess.run(tf.add_n([self.model_loss_weights[i]*l.loss_val(self.model.outputs[i]) for i in range(len(self.model.outputs)) if i != constraint['loss']]), feed_dict = {self.input_tensor:batch_data})
                    #print("batch lagr mult ", lm_avg[-1], " lagr loss ", lagr_avg[-1], "+ ", others, " = total ", total_avg[-1])
                #print("Epoch ", str(i), ": ", end ="")
                for loss_layer in epoch_avg.keys():#self.model.outputs:
                    #epoch_loss = np.sum(epoch_avg[loss_layer.name])
                    epoch_loss = np.mean(epoch_avg[loss_layer])  
                    self.hist[loss_layer].append(epoch_loss)
                    if self.verbose:
                        print(loss_layer.split("/")[0], " : ", epoch_loss, " \t ", end="")
                
                if self.verbose:
                    print( "lagr ", np.mean(lm_avg))
                    print()
                for i in range(len(lagr_vars)):
                    self.hist[lagr_vars[i].name].append(np.mean(lm_avg))
                #print(" Lagr mult for loss ", i, ": ", np.mean(lm_avg), "\t ", end="")
                #print(" Lagr Loss : ", np.mean(lagr_avg), " \t", end ="")
                #print(" Total Loss : ", np.mean(total_avg), " \t", end ="")
                #print()
        #Callbacks?

                #summary, loss = result[1], result[2]
                #writer.add_summary(summary, i)

                # if x_val is not None:
                #     assert len(x_val) == self.batch, "Must compare with batches of equal size"
                #     x_val = load_data(x_val)
                #     summary, val_loss = sess.run([summary_val, self.loss],
                #                                       feed_dict={self.input_tensor: x_val})
                #     writer.add_summary(summary, i)
                # else:
                #     val_loss = np.nan

                #if self.verbose:
                    #t = time.time()
                    #print('{}/{}, Loss:{:0.3f}, Val:{:0.3f}, Seconds: {:0.1f}'.
                    #      format(i, self.epochs, loss, val_loss, t - t0))
                    #if i % 1000 == 999:
                        #print('Saving at {} into {}'.format(i, self.log_dir))
                        #saver.save(sess, os.path.join(self.log_dir, "model_{}.ckpt".format(i)))

    def anneal_loss_weights(self):
        anneal_lw=[]
        for lw in range(len(self.model_loss_weights)):
            if (lw) in [int(i) for i in list(self.anneal_functions.keys())]:
                loss_weight_tensor = tf.Variable(self.anneal_functions[lw](1),  trainable = False, name ='annealed_weight_'+str(lw))
                #loss_weight_layer.trainable = False
                self.model_loss_weights[lw] = loss_weight_tensor
                anneal_lw.append(loss_weight_tensor)
        return anneal_lw


    def build_mismatch_model(self, config):
        self.mm = NoiseModel(dataset = self.dataset, config = config)
        self.mm.encoder_layers = self.mm._build_architecture([self.input], encoder = True)
        self.mm.output_layer = self._decoder()(self.mm.encoder_layers[len(self.mm.encoder_layers)-1]['act'])
            
            # want it to be the operation / keras model, called on 
        recon_loss = Loss(**{'type': self.recon,
                                    'layer': -1,
                                    'encoder': False,
                                    'weight': 1
                                })
        self.mm.loss_functions = recon_loss.make_function()
        output = [self.recon_true, self.mm.output_layer]
        self.mm.loss_outputs = self.mm.loss_functions(output)
        self.mm.model_losses = l.dim_sum #(self.mm.loss_outputs)
        self.mm.loss_weights = 1.0

        self.mm.model = keras.models.Model(inputs = [self.input_tensor], outputs = [self.mm.loss_outputs])
        self.mm.model.compile(optimizer = self.optimizer, loss = [self.mm.model_losses], loss_weights = [self.mm.loss_weights]) # metrics?
        print(self.mm.model.summary())
        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()



    def get_generator(self):
        pass

    def get_encoder(self, encoder_args):
        pass

    def get_loss_history(self):
        pass


    def save_model(self, filename, model = None):
        if model is None:
            model = self.model
        json_string = model.to_json()
        open(filename, 'w').write(json_string)


    def save_weights(self, filename):
        self.model.save_weights(filename+'_weights', overwrite=True)


    def pickle_dump(self):
        fle = open(str(self.filename+".pickle"), "wb")
        stats = {}
        #stats['mi'] = model.mi
        #stats['tc'] = model.tc
        for k in self.hist.keys():
            print("Dumping history ", k, " with length ", len(self.hist[k]))
            #if 'screening' in k or 'info_dropout' in k or 'vae' in k or 'noise_loss' in k:
            stats[k] = self.hist[k]
        for k in self.test_results.keys():
            stats[str('test_'+k)] = self.test_results[k]
        pickle.dump(stats, fle)