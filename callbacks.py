from keras.callbacks import Callback

class BetaCallback(Callback):
    def __init__(self, functions, layer_inds):#betas, sched, screening = False, recon = False): # mod, minbeta=0):
        self.layer_inds = layer_inds
        self.layer_names = ['anneal_'+str(l) for l in layer_inds]
        self.anneal_functions = functions
        #if recon:
        #    self.model.get_layer('recon').set_weights([np.array([self.betas[0]])])
        #else:
        #    self.model.get_layer('betas').set_weights([np.array([self.betas[0]])])
        #if screening:
        #    self.model.get_layer('recon').set_weights([np.array([1-self.betas[0]])])
        #self.k = 1

    def on_epoch_begin(self, epoch, logs={}):
        for l in range(len(self.layer_inds)): 
            self.model.get_layer(self.layer_names[l]).set_weights([np.array([self.anneal_functions[l](epoch)])])
        #if epoch == self.sched[self.k]:
        #    if self.recon:
        #        self.model.get_layer('recon').set_weights([np.array([self.betas[self.k]])])
        #    else:
        #        self.model.get_layer('betas').set_weights([np.array([self.betas[self.k]])])
        #    if self.screening:
        #        self.model.get_layer('recon').set_weights([np.array([1-self.betas[self.k]])])
        #    if self.k < len(self.sched)-1:
        #        self.k = self.k+1