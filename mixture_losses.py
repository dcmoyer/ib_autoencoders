import sys
#import utils
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import objectives
from keras.layers import Lambda, merge
from keras.callbacks import Callback, TensorBoard
#import layers
#import losses as l

''' Mutual Information Estimators
for Corex?  mixing binary and gaussian / ln'''
# USE THIS DIRECTLY AS LAMBDA loss
def gaussian_kl(inputs):
    [mu1, logvar1, mu2, logvar2] = inputs
    return .5*logvar2-.5*logvar1 + tf.divide(K.exp(logvar1) + (mu1 - mu2)**2, 2*K.exp(logvar2)+K.epsilon()) - .5

def binary_kl(inputs):
    [mu1, mu2] = inputs
    mu1 = K.clip(mu1, K.epsilon(), 1-K.epsilon())
    mu2 = K.clip(mu2, K.epsilon(), 1-K.epsilon())
    return tf.multiply(mu1, K.log(mu1)) + tf.multiply(1-mu1, K.log(1-mu1))- tf.multiply(mu1, K.log(mu2)) - tf.multiply(1-mu1, K.log(1-mu2))


def logsumexp(mx, axis, keepdims = False):
    #return tf.reduce_logsumexp(mx, axis = axis)
    cmax = K.max(mx, axis=axis)
    cmax2 = K.expand_dims(cmax, 1)
    mx2 = mx - cmax2
    return cmax + K.log(K.sum(K.exp(mx2), axis=1, keepdims = keepdims))

# to incorporate ECHO noise, modify stats / calculation of KL (q(z|x(k)) vs. q(z|x(l)))
def mixture_tc_gaussian(inputs, maximize = False): #mu2, logvar2,
    [mu1, logvar1] = inputs
    if maximize:
        # could easily do max  H(Zj) - min H(Z) 
        raise NotImplementedError('maximize mixture tc')
    else:
        return mixture_mi_gaussian([mu1, logvar1], marginals = True, maximize = False) - mixture_mi_gaussian([mu1, logvar1], marginals = False, maximize = True)
        #return mixture_mi_gaussian(mu1, logvar1, mu2, logvar2, marginals = True, maximize = False) 
        #       - mixture_mi_gaussian(mu1, logvar1, mu2,  logvar2, marginals = False, maximize = True)

def cross_mixture_ent(inputs, marginals = False, maximize = False):
    [mu1, logvar1, mu2, logvar2] = inputs
    #mi = mixture_mi_gaussian(mu1, logvar1, mu2, logvar2,  marginals = marginals, maximiz e = maximize)
    #xent = 
    pass

''' Mixture MI estimators '''
def mixture_mi_gaussian(inputs, mu2 = None, logvar2 = None, marginals = False, maximize = False):
    '''Kolchinsky & Tracey estimators of mutual information : mixture of gaussians'''
    # mu2, logvar2 here to support cross entropy estimation (as second arg)
    if len(inputs) == 4:
        [mu1, logvar1, mu2, logvar2] = inputs
    else:
        [mu1, logvar1] = inputs

    if mu2 is None or logvar2 is None:
        mu2 = mu1
        logvar2 = logvar1

    # input = z_mean, z_var
    mu1 = K.expand_dims(mu1, 1)
    mu2 = K.expand_dims(mu2, 0) 
    logvar1 = K.expand_dims(logvar1, 1)
    logvar2 = K.expand_dims(logvar2, 0)
    
    #kl calculation
    if maximize:
        # works only for marginals = True
        divergences = _bhatta_gaussian(mu1, logvar1, mu2, logvar2, marginals = marginals)
    else: # minimize
        print("Gaussian KL Mixture Estimation (Minimizing MI)")
        divergences = gaussian_kl([mu1, logvar1, mu2, logvar2])
        #divergences = K.sum(divergences, axis = -1) #gaussian kl needs to be summed over independent dim
    
    n = mu1.get_shape().as_list()[0] 
    result = _mi_from_div(divergences, n, marginals)
    
    return K.expand_dims(result,1) if not marginals else result

def mixture_mi_binary(inputs, maximize = False, marginals = False):
    if len(inputs)== 1:
        [m1] = inputs
        m2 = m1
    else:
        [m1, m2] = inputs

    mu1 = K.expand_dims(m1, 1)
    mu2 = K.expand_dims(m2, 0)
    if maximize:
        divergences = _chernoff_binary(mu1, mu2, alpha = .5, marginals = marginals)
    else:
        divergences = binary_kl([mu1, mu2])

    n = mu1.get_shape().as_list()[0]
    result = _mi_from_div(divergences, n, marginals)
    return K.expand_dims(result,1)


def cross_mixture_mi():
    raise NotImplementedError
    # would be important for E_Q* [sum q(z|x)]

def _mi_from_div(divergences, n, marginals):
    ''' mixture estimator : log sum (avg really) exp of -divergences'''

    # for joint div, sum over marginal div (within exponent of estimator)
    divergences = divergences if marginals else K.sum(divergences, axis = -1) 
    
    batch_size = K.cast(K.shape(divergences)[0], divergences.dtype)  # This is a node tensor, so we can't treat as integer
    avg_log = Lambda(lambda x: x - K.log(batch_size))

    mi_est = -1*(avg_log(logsumexp(-divergences, axis = 1)))# - K.log(n*1.0))
    
    # sum over marginals to get one qty
    mi_est = K.sum(mi_est, axis = -1, keepdims = True) if marginals else mi_est
    return mi_est

def _bhatta_gaussian(mu1, logvar1, mu2, logvar2, marginals = False):
    if marginals:
        bhatta = .25*K.log(.25*(K.exp(logvar1 - logvar2)+K.exp(logvar2 - logvar1)+2)) + tf.divide(.25*(mu1-mu2)**2) / (K.exp(logvar1) + K.exp(logvar2) + K.epsilon())
    else:
        # USED IN ANY MAXIMIZATION OF MIXTURE MI / ENT (i.e. minimizing TC)
        mean_cov = .5*(K.exp(logvar1)+K.exp(logvar2)) # need to account for being 3d
        #mean_prec = tf.matrix_inverse(mean_cov)
        
        #(mu1-mu2).T prec(mean_cov(which is diagonal)) (mu1-mu2) + .5 ln (det mean cov) - .25 ln (det cov1 det cov2)    
        bhatta = .125* K.sum(tf.divide((mu1 - mu2)**2, mean_cov + K.epsilon()), axis = -1) #*det(mean)/sqrt(det1*det2))
        bhatta = bhatta + .5*K.sum(K.log(mean_cov + K.epsilon()), axis = -1) - .25*K.sum(logvar1, axis = -1) - .25*K.sum(logvar2, axis = -1)
    return bhatta

def _bhatta_binary(mu1, mu2, marginals = True):
    return _chernoff_binary(mu1, mu2, alpha = .5, marginals = marginals)

def _chernoff_binary(mu1, mu2, alpha = .5, marginals = True):
    mu1 = K.log(mu1) # clip
    mu2 = K.log(mu2) # clip

    # Chernoff distance = sum ( p(x)^alpha * q(x)^(1-alpha) ), calculated via logs
    if marginals:
        chernoff = K.exp(alpha*mu1 + (1-alpha)*mu2) + K.exp(alpha*(1-mu1) + (1-alpha)*(1-mu2))
        chernoff = K.sum(chernoff, axis = -1)
    else:
        # does this make sense?  p(x) = prod p(xi)
        chernoff = K.exp(alpha*K.sum(mu1, axis = -1) + (1-alpha)*K.sum(mu2, axis = -1)) + K.exp(alpha*(K.sum(1-mu1, axis = -1)) + (1-alpha)*(K.sum(1-mu2, axis = -1)))
    return -K.log(chernoff)

def _chernoff(alpha, binary = False):
    raise NotImplementedError