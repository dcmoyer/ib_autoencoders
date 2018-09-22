import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import objectives
import keras.losses
from keras.layers import Lambda, Concatenate, average, concatenate, add
from keras.callbacks import Callback, TensorBoard
sys.path.insert(1, '/Users/brekels/autoencoders/')
import layers
from functools import partial
import mixture_losses as mix
from sklearn.metrics import log_loss

EPS = K.epsilon()
# EVERYTHING RETURNS BATCH X value AS DEFAULT (may sum or average if see fit)
    # return_dimensions option?

def dim_sum(true, tensor, keepdims = False):
    #print('DIMSUM TRUE ', _true)
    #print('DIMSUM Tensor ', tensor)
    return K.sum(tensor, axis = -1, keepdims = keepdims)

def dim_sum_one(tensor, keepdims = False):
    #print('DIMSUM TRUE ', _true)
    #print('DIMSUM Tensor ', tensor)
    return K.sum(tensor, axis = -1, keepdims = keepdims)

def identity(true, tensor):
    return tensor

def identity_one(tensor):
    return tensor

def loss_val(tensor):
    return K.mean(K.sum(tensor, axis = -1), axis = 0)

def sum_all(tensor):
    return K.sum(tensor)

def logsumexp(x, axis = -1, keepdims = False):
    #return tf.reduce_logsumexp(mx, axis = axis)
    m = K.max(x, axis=axis, keepdims = True) # keep dims for broadcasting
    return m + K.log(K.sum(K.exp(x - m), axis=axis, keepdims=keepdims)) + K.epsilon()

def logmeanexp(x, axis = -1, keepdims = False):
    m = K.max(x, axis=axis, keepdims= True)
    return m + K.log(K.mean(K.exp(x - m), axis=axis, keepdims=keepdims)) + K.epsilon()


def bir(inputs):
    z_mean, z_logvar = inputs
    print("BIR VAR SHAPE ", z_logvar)
    dim = tf.to_float(tf.shape(z_mean)[-1])
    return -.5*dim*z_logvar # will be summed over dim later

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def mmd_loss(inputs, kernel = 'gaussian', gaussian = True, d=500, gamma = 1.0):
    print("INPUTS: ", inputs)
    print("mmd ")
    if not isinstance(inputs, list):
        q = inputs
        p = tf.random_normal(tf.shape(q))
    elif len(inputs) == 1:
        q = inputs[0]
        p = tf.random_normal(tf.shape(q))
    else:
        q, p = inputs
    
    if kernel == 'gaussian':
        q_kernel = compute_kernel(q, q)
        p_kernel = compute_kernel(p, p)
        qp_kernel = compute_kernel(q, p)
        mmd = tf.reduce_mean(q_kernel) + tf.reduce_mean(p_kernel) - 2 * tf.reduce_mean(qp_kernel)
        return tf.expand_dims(tf.expand_dims(mmd, 0), 1)
    else: #if kernel == 'random': 
        
        W = tf.random_normal((K.int_shape(q)[-1], d))
        phi_Wq = tf.sqrt(2/gamma) * tf.matmul(q,W) + tf.transpose(2*np.pi*tf.random_uniform((d,1)))
        phi_Wq = tf.sqrt(2/d) * tf.cos(phi_Wq)
        phi_Wp = tf.sqrt(2/gamma) * tf.matmul(p,W) + tf.transpose(2*np.pi*tf.random_uniform((d,1)))
        phi_Wp = tf.sqrt(2/d) * tf.cos(phi_Wp)
        
        mmd = K.mean((phi_Wq - phi_Wp), axis = 0, keepdims = True)**2
        return K.sum(mmd, axis = -1, keepdims = True) 

def made_marginal_density(inputs, method = 'maf'):
    # target - true ... output layer for MADE
    # ONLY WORKS FROM ASSUMED indpt std Gaussian NOISE MODEL
    gaussian_vals, logvars = inputs
    #target, [log_vars, rand_inputs] = inputs
    if method in ['maf', 'gaussian']:
        normal_pdf = ((2*np.pi)**-.5)*K.exp(-.5*gaussian_vals**2)
        sum_jacobian = logvars
        #add([K.sum(-.5*log_var, axis = -1) for log_var in list_of_log_vars])
        log_density = normal_pdf + sum_jacobian
        # evaluate density of rand_inputs
        # sum jacobian
        # each target value against some random input?
    print("LOG DENSITY ", K.int_shape(log_density))
    return log_density
    # evaluate MADE at inverse of transform of z?


def echo_var(inputs, d_max = 50):
    print("INPUTS: ", inputs)
    if isinstance(inputs, list):
        cap_param = inputs[0]
    else:
        cap_param = inputs

    min_capacity = 16.0 / d_max # d_max ... don't have access to d_max to actually pass
    # compare to what Greg's implementation calculates for D, s.b. easy to verify

    #capacities = tf.identity(tf.nn.softplus(-cap_param) - np.log(self.c_min), name='capacities')
    capacities = tf.nn.softplus(- cap_param) #tf.identity(tf.nn.softplus(- cap_param), name='capacities') #tf.maximum(tf.nn.softplus(- cap_param), min_capacity, name='capacities')
    #cap = tf.reduce_sum(capacities, name="capacity")
    print("Capacities ", capacities)
    cap = K.var(capacities, axis = 0, keepdims = True)
    return tf.expand_dims(cap, 1) #tf


def echo_minmax(inputs, _min = True, d_max = 50):
    if isinstance(inputs, list):
        cap_param = inputs[0]
    else:
        cap_param = inputs

    min_capacity = 16.0 / d_max # d_max ... don't have access to d_max to actually pass
    # compare to what Greg's implementation calculates for D, s.b. easy to verify

    #capacities = tf.identity(tf.nn.softplus(-cap_param) - np.log(self.c_min), name='capacities')
    capacities = tf.nn.softplus(- cap_param) #tf.identity(tf.nn.softplus(- cap_param), name='capacities') #tf.maximum(tf.nn.softplus(- cap_param), min_capacity, name='capacities')
    #cap = tf.reduce_sum(capacities, name="capacity")
    print("Capacities ", capacities)
    if _min:
        cap = K.min(cap_param, axis = 0, keepdims = True)
    else:
        cap = K.max(cap_param, axis = 0, keepdims = True)
    return tf.expand_dims(cap, 1) #tf

def echo_loss(inputs, d_max = 50):
    echo_layer = inputs[0]
    cap_param = echo_layer #.get_weights()[0]#get_cap_param()
    capacities = tf.nn.softplus(- cap_param) #tf.identity(tf.nn.softplus(- cap_param), name='capacities') #tf.maximum(tf.nn.softplus(- cap_param), min_capacity, name='capacities')
    cap = tf.reduce_sum(capacities, name="capacity")
    return tf.expand_dims(tf.expand_dims(cap,0), 1)

# def echo_loss2(inputs, d_max = 50):
#     print("INPUTS: ", inputs)
#     cap_param = inputs[0]
#     min_capacity = 16.0 / d_max # d_max ... don't have access to d_max to actually pass
#     # compare to what Greg's implementation calculates for D, s.b. easy to verify

#     #capacities = tf.identity(tf.nn.softplus(-cap_param) - np.log(self.c_min), name='capacities')
#     capacities = tf.nn.softplus(- cap_param) #tf.identity(tf.nn.softplus(- cap_param), name='capacities') #tf.maximum(tf.nn.softplus(- cap_param), min_capacity, name='capacities')
#     cap = tf.reduce_sum(capacities, name="capacity")
#     return tf.expand_dims(tf.expand_dims(cap,0), 1)


def gaussian_logpdf(eval_pt, mu = 0.0, logvar = 0.0):
    if eval_pt is None:
        eval_pt = mu
    if isinstance(mu, float) or mu == 0.0:
        mu = K.variable(mu)
    if isinstance(logvar, float) or logvar == 0.0:
        logvar = K.variable(logvar)
    
    #dim = K.int_shape(eval_pt)[1]
    var = K.exp(logvar)
    log_pdf = -.5*K.log(2*np.pi) - (logvar / 2.0)- (eval_pt - mu)**2 / (2 * var) 

    return log_pdf

def binary_logpdf(true, pred, binarized = False): # assumes 
    pred = K.clip(pred, K.epsilon(), 1-K.epsilon())
    if binarized:
        masked_prob = tf.where(tf.equal(true, tf.ones_like(true)), pred, 1-pred)
        return tf.log(masked_prob)
    else:
        return tf.multiply(true, tf.log(pred)) + tf.multiply(1-true, tf.log(1-pred))

# mixture estimates mutual information anyway, 
# marginals false is the default?  Otherwise will get a telescoping sum of 
def corex(inputs, recon = None, mi = None, marginals = True, beta = 1.0):
    if len(inputs) == 4:
        print("LENGTH 4 INPUTS")
        [mu, logvar, true, pred] = inputs
    else:
        print("LENGTH ", len(inputs), " INPUTS")
        [mu, logvar, true] = inputs[0:2]
        pred = inputs[3:]

    # feed callable or string referring to method
    if recon is None:
        recon = mse
    if mi is None:
        mi = "mixture"
    if mi == "mixture":
        mi = partial(mix.mixture_mi_gaussian, marginals = marginals)
    elif mi == "gaussian" or mi == "normal":
        mi = gaussian_mi
    elif mi == "lognormal":
        mi = lognormal_mi        
    # recon should be parsed from str to callable in loss_args
    print('recon ', recon, type(recon))
    print('mi callable? ', mi, type(mi))
    # min H(Xi|Z) + H(Zj) - H(Zj|X) (or joint MI)
    loss = dim_sum_one(recon([true, pred]))
    reg = dim_sum_one(mi([mu, logvar]))
    return K.expand_dims(loss + beta*reg, 1)# minimize corex objective


def recon_mse_mi(inputs):

    h_const = tf.constant(0.5 * np.log(2. * np.pi), dtype=tf.float32)
    recon_error = tf.subtract(true, pred, name="recon_error")
    mse = tf.reduce_mean(tf.square(recon_error), axis=0, name='mean_error')
    recon_loss = tf.add(h_const, 0.5 * tf.reduce_sum(tf.log(mse + 1e-5)), name='recon_loss')

# USE THIS DIRECTLY AS LAMBDA loss
def gaussian_kl(inputs):
    [mu1, logvar1, mu2, logvar2] = inputs
    return .5*logvar2-.5*logvar1 + tf.divide(K.exp(logvar1) + (mu1 - mu2)**2, 2*K.exp(logvar2)+K.epsilon()) - .5
    #return K.sum(.5*logvar2-.5*logvar1 + tf.divide(K.exp(logvar1) + (mu1 - mu2)**2,(2*K.exp(logvar2)+K.epsilon())) - .5, axis = -1)

# with defaults?
def gaussian_prior_kl(inputs):
    print("Gaussian prior KL inputs : ", inputs)
    [mu1, logvar1] = inputs
    mu2 = K.variable(0.0)
    logvar2 = K.variable(0.0)
    return gaussian_kl([mu1, logvar1, mu2, logvar2])

def gaussian_mi(inputs, sample = True):
    mu1 = inputs[0]
    logvar1 = inputs[1]
    if sample:
        samples = layers.vae_sample([mu1, logvar1])
    mu2 = K.mean(samples, axis = -1, keepdims = True)
    logvar2 = K.log(K.var(samples, axis = -1, keepdims = True)+EPS)
    return gaussian_kl([mu1, logvar1, mu2, logvar2])

def binary_kl(inputs):
    [mu1, mu2] = inputs
    mu1 = K.clip(mu1, K.epsilon(), 1-K.epsilon())
    mu2 = K.clip(mu2, K.epsilon(), 1-K.epsilon())
    return tf.multiply(mu1, K.log(mu1)) + tf.multiply(1-mu1, K.log(1-mu1))- tf.multiply(mu1, K.log(mu2)) - tf.multiply(1-mu1, K.log(1-mu2))

def binary_crossentropy(inputs):
    print("Binary cross entropy inputs : ", inputs)
    if len(inputs) == 2:
        [mu1, mu2] = inputs
        if isinstance(mu2, list):
            print("**** LOSS AVERAGING BCE ****")
            return average([K.binary_crossentropy(mu1, pred) for pred in mu2])
        else:
            #return -tf.multiply(mu1, tf.log(mu1+10**-7))+10**-10*mu2
            return K.binary_crossentropy(mu1, mu2)
    else:
        true = inputs[0]
        return average([K.binary_crossentropy(true, inputs[pred]) for pred in range(1, len(inputs))])

def bce_np(inputs):
    def np_bce(x, y, eps= 10**-7):
        y = np.clip(y, eps, 1-eps)
        return -np.multiply(x, np.log(y))-np.multiply(1-x, np.log(1-y))

    #print("Binary cross entropy inputs : ", inputs)
    if len(inputs) == 2:
        [mu1, mu2] = inputs
        if isinstance(mu2, list):
            print("**** LOSS AVERAGING BCE ****")
            return np.mean([np_bce(mu1, pred) for pred in mu2])
        else:
            #return -tf.multiply(mu1, tf.log(mu1+10**-7))+10**-10*mu2
            print("NP BCE shape ", np_bce(mu1, mu2).shape)
            return np.mean(np.sum(np_bce(mu1, mu2), axis = -1))
    else:
        true = inputs[0]
        return true

def mean_squared_error(inputs):
    if len(inputs) == 2:
        [mu1, mu2] = inputs
        if isinstance(mu2, list):
            return average([mse(mu1, pred) for pred in mu2])
        else:
            return mse(mu1, mu2)
    else:
        true = inputs[0]
        return average([mse(true, inputs[pred]) for pred in range(1, len(inputs))])

def mse(a, b):
    return (a-b)**2


def _parse_iwae(inputs):
    if len(inputs) == 5:
        [z_mean, z_logvar, z_samples, x_true, x_preds] = inputs
        z_samples = [z_samples]
        x_preds = [x_preds]
    else:
        z_mean = inputs[0]
        z_logvar = inputs[1]
        i = int((len(inputs)-3)/2)
        z_samples = inputs[2:(i+2)]
        x_preds = inputs[i+3:]
        x_true = inputs[i+2]

    return z_mean, z_logvar, z_samples, x_true, x_preds


def importance_weights(inputs, recon = 'bce', prior = gaussian_logpdf, lognormal = False,  beta = 1.0, return_ll = False):
    z_mean, z_logvar, z_samples, x_true, x_preds = _parse_iwae(inputs)

    if lognormal:
        raise NotImplementedError
    else:
       
        assert(len(z_samples) == len(x_preds))
        

        for i in range(len(z_samples)):
            z_sample = z_samples[i]
            x_pred = x_preds[i]

            lprior_all = prior(z_sample)
            #log_prior = K.sum(lprior_all, axis = -1)
            if (recon is None or recon == 'bce' or 'binary' in recon):
                ll_all = binary_logpdf(x_true, x_pred)
            else:
                sigma = .1
                ll_all = gaussian_logpdf(x_true, mu = x_pred, logvar = K.log(sigma*K.ones_like(x_pred)))

            # PDF'S CAN SUM OVER LATENT DIM... just need to concat over k
            #log_likelihood = K.sum(ll_all, axis = -1)
            
            lprop_all = gaussian_logpdf(z_sample, mu = z_mean, logvar = z_logvar)
            #log_proposal = K.sum(log_proposa, axis = -1)
            #log_w = tf.multiply(beta, log_likelihood) + log_prior - log_proposal
            #print("log w  ", ll_all, lprior_all, ' divided by prop ', lprop_all)
            log_w = K.sum(tf.multiply(beta, ll_all), axis = -1) + K.sum(lprior_all, axis = -1) - K.sum(lprop_all, axis = -1)
            ll_samp = K.sum(ll_all, axis = -1)

            iw = K.expand_dims(log_w,1) if i == 0 else concatenate([iw, K.expand_dims(log_w,1)], axis = -1)
            ll = K.expand_dims(ll_samp, 1) if i == 0 else concatenate([ll, K.expand_dims(ll_samp, 1)], axis = -1)

        if return_ll:
            #print('iw, ', iw, ' ll', ll)
            return iw, ll
        else:
            return iw

def iwae_loss(inputs, recon = 'bce', lognormal = False, prior = gaussian_logpdf, beta = 1.0):
    print("**** beta? first arg tensor **** ", inputs[0], inputs[0].name, inputs[0].shape[0],  inputs[0].shape[1])
    if 'beta' in inputs[0].name or (inputs[0].shape[0]== 1 and inputs[0].shape[1]== 1):
       beta = inputs[0]
       inputs = inputs[1:]
    #    beta = beta # starts with stat layer
    #else:
    #    beta = inputs[0]
    #    inputs = inputs[1:]
    iw = importance_weights(inputs, recon = recon, prior = gaussian_logpdf, lognormal = lognormal, beta = beta)     
    return -K.mean(logmeanexp(iw, axis = -1, keepdims = True), axis=0, keepdims = True) # -1 axis = k samples


def niw_entropy(inputs, recon = 'bce', lognormal = False, prior = gaussian_logpdf, beta = 1.0):
    beta = inputs[0]
    inputs = inputs[1:]
    iw, ll = importance_weights(inputs, recon = recon, prior = gaussian_logpdf, lognormal = lognormal, beta = beta, return_ll = True)
    
    #n_iw = tf.divide(iw, K.epsilon() + K.sum(iw, axis = -1, keepdims = True))

    normalizer = logsumexp(iw, axis = -1, keepdims = True)#logsumexp(iw, axis = -1, keepdims = True)
    log_n_iw = iw - normalizer
    n_iw = K.exp(log_n_iw)
    #print('IW EXP ONLY: return mult to be avgd ', tf.multiply(n_iw, ll), ' ll shape ', ll)
    #iwae_b = logmeanexp(iw, axis = -1) # -1 axis = k samples
    return -tf.multiply(n_iw, log_n_iw)

def iw_pxz(inputs, recon = 'bce', lognormal = False, prior = gaussian_logpdf, beta = 1.0):
    beta = inputs[0]
    inputs = inputs[1:]
    iw, ll = importance_weights(inputs, recon = recon, prior = gaussian_logpdf, lognormal = lognormal, beta = beta, return_ll = True)
    
    #n_iw = tf.divide(iw, K.epsilon() + K.sum(iw, axis = -1, keepdims = True))
    normalizer = logsumexp(iw, axis = -1, keepdims = True)
    n_iw = K.exp(iw - normalizer)
    
    #print('IW EXP ONLY: return mult to be avgd ', tf.multiply(n_iw, ll), ' ll shape ', ll)
    #iwae_b = logmeanexp(iw, axis = -1) # -1 axis = k samples
    return -K.sum(tf.multiply(n_iw, ll), axis = -1, keepdims = True)# + iwae_b) # minimize Eq*logp(x|z) + Ex log p_beta(x)

def beta_gradient(inputs, recon = 'bce', lognormal = False, prior = gaussian_logpdf):
    beta = inputs[0]
    inputs = inputs[1:]
    z_mean, z_logvar, z_samples, x_true, x_preds = _parse_iwae(inputs)
    k = len(z_samples) if isinstance(z_samples, list) else 1
    batch = K.int_shape(z_mean)[0]
    #print("BETA ", beta)
    #print('beta grad inputs ', inputs)
    iw, ll = importance_weights(inputs, recon = recon, prior = gaussian_logpdf, lognormal = lognormal, beta = beta, return_ll = True)
    # shape of iw = (batch, k samples)
    #n_iw = tf.divide(K.exp(iw), K.epsilon()+K.exp(normalizer))
    # don't actually need n_iw since taking expectation wrt Wi

    # sum over k samples, result is shape (batch, )
    #Eqstar_logp = K.sum(tf.divide(tf.multiply(K.exp(iw), ll), (K.epsilon() + K.sum(K.exp(iw), axis = -1, keepdims = True))), axis = -1) # average over k samples of log_likelihood (sum (wi * ll) / sum(wi))

    print('iw ', iw)
    print('ll ', ll)
    print('BETA GRAD ACCORDING TO IW ONLY')
    #normalized iw
    normalizer = logsumexp(iw, axis = -1, keepdims = True)
    n_iw = K.exp(iw-normalizer)
    Eqstar_logp = K.sum(tf.multiply(n_iw, ll), axis = -1, keepdims = True)


    # sum ?
    #Eqstar_logp = 
    #Eqstar_logp = K.exp(logsumexp(iw + K.log(-ll), axis = -1, keepdims = True))
    #Eqstar_logp = K.mean(tf.multiply(K.exp(iw), ll), axis = -1, keepdims = True)

    inp_list = [x_true]
    inp_list.extend(x_preds)
    if len(x_preds)==1:
        Eq_logp = K.sum(binary_logpdf(x_true, x_preds[0]), axis = -1, keepdims = True)
    else:
        Eq_logp = K.sum(average([binary_logpdf(x_true, x_pred) for x_pred in x_preds]), axis = -1, keepdims = True)

    #dl_db = tf.multiply(1/(batch*k)*dw, 1 + K.log(n_iw) - log_prior - beta*log_likelihood) + Eq_logp # - log N*k for average in n_iw?
    beta_grad = Eq_logp - Eqstar_logp
    #K.expand_dims(Eqstar_logp, 1)
    return beta_grad #K.mean(beta_grad, axis = 0)



def beta_gradient_old(inputs, recon = 'bce', lognormal = False, prior = gaussian_logpdf):
    beta = inputs[0]
    inputs = inputs[1:]
    z_mean, z_logvar, z_samples, x_true, x_preds = _parse_iwae(inputs)
    k = len(z_samples) if isinstance(z_samples, list) else 1
    batch = K.int_shape(z_mean)[0]

    iw, [log_likelihood, log_prior, log_proposal] = importance_weights(inputs, recon = recon, prior = gaussian_logpdf, lognormal = lognormal, beta = beta, return_all = True)
    # already batch x k ? 
    iw = K.reshape(iw, (k, batch))
    
    # SHOULD I BE NORMALIZING OVER SOME OR ALL?
    n_iw = tf.divide(iw, K.sum(iw, axis = 0)) # average over k

    #n_iw = tf.divide(iw, K.sum(iw))  gaussian_logpdf(z_samples, z_mean, z_logvar)
    log_likelihood = K.reshape(log_likelihood, (k, batch))
    mean_ll = K.sum(tf.multiply(iw, log_likelihood), axis = 0)/K.sum(iw, axis = 0) # average over k samples of log_likelihood (sum (wi * ll) / sum(wi))

    dw = tf.multiply(n_iw, log_likelihood - mean_ll) # REDUNDANT CALCULATION WITH IW
    
    inp_list = [x_true]
    inp_list.extend(x_preds)
    Eq_logp = binary_crossentropy(inp_list)

    dl_db = tf.multiply(1/(batch*k)*dw, 1 + K.log(n_iw) - log_prior - beta*log_likelihood) + Eq_logp # - log N*k for average in n_iw?
    return dl_db

    #     z_mean_rep = merge([z_mean]*k, mode = 'concat', concat_axis = 0) 
    #     z_noise_rep = merge([z_noise]*k, mode = 'concat', concat_axis = 0)
    #     x_tru = merge([x_tru]*k, mode = 'concat', concat_axis = 0)
    #     log_proposal = log_prob_ind(z_mean_rep, z_noise_rep, evaluated_at = z_samples)
        
    #     # INCORRECT : REPLACE FOR BINARY MNIST
    #     log_p_xi = K.sum(K.log(K.clip(x_decode, K.epsilon(), 1-K.epsilon())), axis = -1, keepdims = True)
    #     #log_q_z = log_prob_ind(K.zeros_like(z_samples), K.ones_like(z_samples), evaluated_at = z_samples)
    #     log_q_z = log_prob_ind(K.mean(z_samples, axis = -1, keepdims = True), K.var(z_samples, axis = -1,keepdims = True), evaluated_at = z_samples)
        
    #     #beta*  +log_q_z -
    #     imp_weights = K.exp(beta*log_p_xi + log_q_z -log_proposal) 
    #     #imp_weights = K.exp(beta*log_p_xi +log_q_z -log_proposal) #)##beta*log_p_xi )#log_p_xi+log_q_z)# - log_proposal) #log_q_z 
    #     #BEST code: imp_weights = K.exp(beta*log_p_xi  + log_q_z - log_proposal)#
    #     #if K.int_shape(imp_weights)[0] is not None:
        
    #     normalized_weights = K.reshape(imp_weights, (k, -1))#K.int_shape(imp_weights)[0]/k))
    #     print('*** imp shape *** ', normalized_weights.shape)
    #     normalized_weights = tf.divide(normalized_weights, K.sum(normalized_weights, axis = 0, keepdims = True))
    #     normalized_weights = K.reshape(normalized_weights, (-1,))
    #     #imp_weights = K.flatten(imp_weights)  

    #     # TO DO: sampling from normalized weights to back propagate (for each data sample)
    #     #important_sample = tf.multinomial(imp_weights, -1)

    #     if binary:
    #         cross_ent = -K.mean(K.binary_crossentropy(x_tru, x_decode), axis = -1)
    #         #likelihood = K.mean(K.log(K.clip(x_decode, K.epsilon(), 1-K.epsilon())), axis = -1)
    #         if both_expectations:
    #             overall_likelihood = neg_ent_x(x_decode, beta = beta)
    #         #print("****cross ent shape: **** ", K.int_shape(cross_ent))


    #     if beta_decode:
    #         conditional_gap = tf.multiply((imp_weights-1), cross_ent)
    #         #conditional_gap = tf.multiply((1-imp_weights), likelihood) #beta*
    #     else:
    #         conditional_gap = beta*tf.multiply((imp_weights-1), cross_ent)
    #         #conditional_gap = beta*tf.multiply((1-imp_weights), likelihood)

    #     if both_expectations:
    #         overall_gap = tf.multiply((1-imp_weights), overall_likelihood)
    #         return K.mean(conditional_gap-overall_gap)
    #     else:
    #         return K.mean(conditional_gap)
    # return importance_loss







def gaussian_entropy(inputs):#logvar = 0.0):
    '''calculates (conditional) entropy per sample, with logvar summed over dimensions '''
    if not isinstance(inputs, list) or len(inputs) == 1:
        logvar = inputs
    else:
        [mu, logvar] = inputs
    return .5*(np.log(2*np.pi)+1+K.sum(logvar, axis = -1, keepdims = True))

def lognormal_entropy(inputs):#mean, logvar):
    [mu, logvar] = inputs
    return K.sum(mu, axis = -1) + gaussian_entropy(logvar)
    # as function of log mean, logvar of log




    '''wrapper for univariate gaussian kl divergence
            specify empirical averaging by mu2 or logvar2 = None
            INFO DROPOUT:   uses this, with z_mean = ln(X), z_logvar = var(ln(X))
                            paper KL divergence is mean 0, logvar empirical (wrong?)
    '''
def gaussian_kl_prior(x_true, merged_decode):
    z_mean, z_logvar = _process_merged(merged_decode)
    return gaussian_prior_kl(z_mean, z_logvar)

# info dropout adjusted _ kl uses this
def gaussian_kl_empirical(x_true, merged_decode):
    z_mean, z_logvar = _process_merged(merged_decode)
    mu2 = K.mean(z_mean, axis = -1, keepdims = True)
    logvar2 = K.log(K.var(z_mean, axis = -1, keepdims = True)+EPS)
    return gaussian_kl(z_mean, z_logvar, mu2, logvar2)

# info dropout (from the paper ) = 0 mean, empirical var
def ido_paper(x_true, merged_decode):
    z_mean, z_logvar = _process_merged(merged_decode)
    logvar2 = K.log(K.var(z_mean, axis = -1, keepdims = True)+EPS)
    return gaussian_kl(z_mean, z_logvar, K.variable(0.0), logvar2)


def _process_merged(merged):
    m = int(merged._keras_shape[-1]/2)
    z_mean = merged[:, m:] 
    z_logvar = merged[:, :m]
    return z_mean, z_logvar



def tc_sampling(inputs, samples = None, k = 100, n = 50000, lognormal = False):
    [mu, logvar] = inputs
    batch = K.cast(K.shape(mu)[0], mu.dtype)
    subtract_log_nb = Lambda(lambda x: x - K.log(n*1.0) - K.log(batch))
    
    # take multiple different samples from z_noise, z_mean rather than just z_act 
    if samples is None:
        for i in range(k):
            samples = layers.vae_sample(mu, logvar) if samples is None else Concatenate(axis = 0)([samples, layers.vae_sample(mu, logvar)])
    samples = K.expand_dims(samples, 1)

    prob_qz_x = _sample_prob_over_batch(samples, mu, logvar, lognormal = lognormal)
    q_z = K.mean(K.log(K.sum(prob_qz_x, axis = -1)) - K.log(n*batch*1.0), axis = 0)
    q_zj = K.sum(K.mean(K.log(prob_qz_x) - K.log(n*batch*1.0), axis = 0), axis = -1)
    return q_z - q_zj

def mi_sampling(inputs, marginals = False, samples = None, k = 100,  n = 50000, lognormal = False):
    [mu, logvar] = inputs
    batch = K.cast(K.shape(mu)[0], mu.dtype)
    subtract_log_nb = Lambda(lambda x: x - K.log(n*1.0) - K.log(batch))
    #mu = K.expand_dims(mu, 0)
    #logvar = K.expand_dims(logvar, 0)
    
    # take multiple different samples from z_noise, z_mean rather than just z_act 
    #for i in k:
    #    pass
        #layers.vae_sample(mu, logvar)
        #samples = K.expand_dims(samples, 1)
    if samples is None:
        for i in range(k):
            samples = layers.vae_sample([mu, logvar]) if samples is None else Concatenate(axis = 0)([samples, layers.vae_sample([mu, logvar])]) 
    samples = K.expand_dims(samples, 1)

    print("SHAPES")
    print("*** samples ", samples, "***")
    print('*** mu ', mu, '***')
    print('*** logvar ', logvar, '***')
    prob_qz_x = _sample_prob_over_batch(samples, mu, logvar, lognormal = lognormal)
    if marginals:
        h_zj_x =  K.mean(gaussian_entropy(logvar) if not lognormal else lognormal_entropy(mu, logvar), axis = 0)
        h_zj = -K.sum(K.mean(subtract_log_nb(K.log(prob_qz_x)), axis = 0), axis = -1)
        return K.expand_dims(h_zj - h_zj_x, 1)
    else:
        h_z = -K.mean(subtract_log_nb(K.log(prob_qz_x)), axis = 0, keepdims = True) # batch mean
        h_zj_x = K.mean(gaussian_entropy(logvar) if not lognormal else lognormal_entropy(mu, logvar), axis = 0, keepdims = True)
        #print("JOINT MI", K.mean(h_z-h_zj_x, axis = 0, keepdims = True))
        return K.expand_dims(h_z - h_zj_x, 1) # - q_z


def _sample_prob_over_batch(samples, mu, logvar, lognormal = False):
    #gauss = tf.contrib.distributions.Normal(0.0, 1.0)
    # assuming samples = k x batch x dim
    mu = K.expand_dims(mu, 0)
    logvar = K.expand_dims(logvar, 0)
    # log normal the same or scaled by 1/x?
    if lognormal:
        print("**** Careful, LOGNORMAL ****  (check density eval) ")
        # check this
        probs = tf.divide(gauss.prob(tf.divide((samples - mu), K.exp(logvar/2)+K.epsilon())), z_act_ext+K.epsilon()) 
    else:
        probs = gaussian_logpdf(samples, mu, logvar)
        #gauss.prob(tf.divide((samples - mu), K.exp(logvar/2)+K.epsilon))
    print("probs ", probs)
    # sum log probability over dimensions, then take sum of probabilities across batch (all other data pts)
    sum_prob_over_batch = K.sum(K.exp(K.sum(probs, axis = -1)) + K.epsilon(), axis = 1)
    return sum_prob_over_batch