from keras.callbacks import Callback
import tensorflow as tf
from keras.backend import get_session
from keras.backend import eval
from collections import defaultdict
class BetaCallback(Callback):
    def __init__(self, functions, layers):#betas, sched, screening = False, recon = False): # mod, minbeta=0):
        self.layers = layers
        self.anneal_functions = functions
    
    def on_epoch_begin(self, epoch, logs={}):
        for l in range(len(self.layers)): 
            tf.assign(self.layers[l],self.anneal_functions[l](epoch)).eval(session=get_session())
            #print("anneal value ", self.layers[l])
            

class DensityTrain(Callback):
    def __init__(self, model, loss_names, outputs, losses, weights = None, lr = .0003):#betas, sched, screening = False, recon = False): # mod, minbeta=0):
        self.model = model
        self.loss_names = loss_names
        self.outputs = outputs
        self.losses = losses
        #self.weights = weights
        self.trainers = {}
        print("TRAINABLES ")
        print()
        #print(tf.trainable_variables())
        print()
        prev = 0

        named_vars = [v for v in tf.trainable_variables() if "masked_autoregressive" in v.name] 
        #print("NAMED VARS for loss:  ", named_vars)
        print("losses ", self.losses)
        print(self.loss_names)
        for l in self.loss_names.keys():
            # ONLY ONE PER
            v = self.loss_names[l]
            loss = self.losses[v-1]
            self.trainers[l] = tf.train.AdamOptimizer(lr).minimize(loss(self.outputs[v-1]), var_list=named_vars)
        
        self.avg = {}
        self.hist = defaultdict(list)
        self.keys = list(self.trainers.keys())
        # for i in self.layer_names.keys():
        #     #learning_rate = tf.placeholder(tf.float32)
        #     named_vars = [v for v in tf.trainable_variables() if i.split("_")[0] in v.name]
        #     print("NAMED VARS for loss: ", i, " : ", named_vars)
        #     print("losses ", self.losses[prev:(self.layer_names[i]-1)])
        #     for l in self.losses[prev:(self.layer_names[i]-1)]:
        #         self.trainers[l] = tf.train.AdamOptimizer(lr).minimize(l, var_list=named_vars)
        #     prev = self.layer_names[i]

        self.x = tf.Variable(0., validate_shape=False)
        self.z = tf.Variable(0., validate_shape=False)

        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()


    def on_epoch_end(self, epoch, logs={}):
        if self.avg:
            for k, v in self.avg:
                self.avg[k] = self.avg[k]/self.num_batches
                self.hist[k].append(self.avg[k])
                print("Epoch: ", epoch, ":    Loss ", k, " : ", v) 
                self.avg[k] = 0
        self.num_batches = 0


    def on_batch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        x = eval(self.x)
        z = eval(self.z)

        try:
            self.num_batches += 1
        except:
            self.num_batches = 0
        # train 
        
        
        
        with self.sess.as_default():
            self.sess.run([self.trainers[k] for k in self.keys], feed_dict={self.model.inputs[0]: x})
            losses = self.sess.run([self.losses[i](self.outputs[i]) for i in range(len(self.outputs))], feed_dict={self.model.inputs[0]: x})
        
        for i in range(len(losses)):
            try:
                self.avg[self.keys[i]] += losses[i]
            except:
                self.avg[self.keys[i]] = losses[i]

class DensityEpoch(Callback):
    def __init__(self, model, loss_names, outputs, losses,  weights = None, batch = 100, lr = .0003):#betas, sched, screening = False, recon = False): # mod, minbeta=0):
        self.model = model
        self.loss_names = loss_names
        self.outputs = outputs
        self.losses = losses
        self.batch = batch
        #self.weights = weights
        self.trainers = {}
        print("TRAINABLES ")
        print()
        #print(tf.trainable_variables())
        print()
        prev = 0

        named_vars = [v for v in tf.trainable_variables() if "masked_autoregressive" in v.name] 
        #print("NAMED VARS for loss:  ", named_vars)
        print("losses ", self.losses)
        print(self.loss_names)
        for l in self.loss_names.keys():
            # ONLY ONE PER
            v = self.loss_names[l]
            loss = self.losses[v-1]
            self.trainers[l] = tf.train.AdamOptimizer(lr).minimize(loss(self.outputs[v-1]), var_list=named_vars)
        
        self.avg = {}
        self.hist = defaultdict(list)
        self.keys = list(self.trainers.keys())
        # for i in self.layer_names.keys():
        #     #learning_rate = tf.placeholder(tf.float32)
        #     named_vars = [v for v in tf.trainable_variables() if i.split("_")[0] in v.name]
        #     print("NAMED VARS for loss: ", i, " : ", named_vars)
        #     print("losses ", self.losses[prev:(self.layer_names[i]-1)])
        #     for l in self.losses[prev:(self.layer_names[i]-1)]:
        #         self.trainers[l] = tf.train.AdamOptimizer(lr).minimize(l, var_list=named_vars)
        #     prev = self.layer_names[i]

        self.x = tf.Variable(0., validate_shape=False)
        self.z = tf.Variable(0., validate_shape=False)

        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()


    def on_epoch_end(self, epoch, logs={}):
        x = eval(self.x)
        z = eval(self.z)
        print("x shape ", x.shape)
        print("z shape ", z.shape)

        n_samples = x.shape[0]
        self.num_batches = floor(n_samples / self.batch)
        self.sess = tf.Session()
        self.hist = defaultdict(list)
        with self.sess.as_default():
            epoch_avg = defaultdict(list)
            total_avg = []
            lagr_avg = []
            lm_avg = []
            perm = np.random.permutation(n_samples)  # random permutation of data for each epoch
            
            


            for offset in range(0, (int(n_samples / self.batch) * self.batch), self.batch):  # inner
                batch_data = x[perm[offset:(offset + self.batch)]]
                self.sess.run([self.trainers[k] for k in self.keys], feed_dict={self.model.inputs[0]: x})
                losses = self.sess.run([self.losses[i](self.outputs[i]) for i in range(len(self.outputs))], feed_dict={self.model.inputs[0]: x})
                
                for i in range(len(losses)):
                    try:
                        self.avg[self.keys[i]] += losses[i]
                    except:
                        self.avg[self.keys[i]] = losses[i]
                
        print("Epoch: ", epoch, ":    Losses ", [(k, self.avg[k]/self.num_batches) for k in self.avg.keys()])
        for k, v in self.avg:
            self.avg[k] = self.avg[k]/self.num_batches
            self.hist[k].append(self.avg[k]) 
            self.avg[k] = 0
        