import pickle 
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
import analysis


folders = [
    'vae_beta',
    'vae_constraint',
    'ido_beta',
    'ido_constraint',
    'echo_add_beta',
    'echo_add_constraint', 
    'echo_add_beta_dim2',
    'echo_mult_beta',
    'echo_mult_constraint',
    'echo_mult_beta_dim2'
]
folders = [str('results/' + f) for f in folders]
#for folder in folders:
#    analysis.rd_curve(folder, beta = 'beta' in folder)

#f = ['results/echoA_small1250_binary_mnist', 'results/echo_mult_small1250_binary_mnist']#, 'results/made_small1250_binary_mnist']
#analysis.rd_curve(f, beta = True, name = 'add vs mult 200')#f[0].split('/')[-1], threshold = .00001)


f = ['results/ido_small1250_binary_mnist', 'results/vae_small1250_binary_mnist',  'results/echoA_small1250_binary_mnist']#, 'results/made_small1250_binary_mnist']
#f = ['results/vae_check_smalltrain_binary_mnist', 'results/echo2add_smalltrain_binary_mnist', 'results/echo_add_smalltrain_Init_binary_mnist']#results/echo_add_smalltrain_Init_binary_mnist']#, 'results/alemi_test_vae_binary_mnist']
analysis.rd_curve(f, beta = True, name = '1250+ido')




#f = [, 'results/']

f = ['results/vae_gated_full_fmnist', 'results/echoAdd_gated_full_fmnist']
analysis.rd_curve(f, beta = True, name = 'gated_full_fmnist')

#f = ['results/echoCONV-smalltrain_binary_mnist', 'results/vaeCONV-smalltrain_binary_mnist']#,'results/echo_alemi_-.5init_binary_mnist']
#analysis.rd_curve(f, beta = True, name = 'conv_smalltrain')


#f = ['results/echo001_conv_smalltrain_fmnist', 'results/vae001_conv_smalltrain_fmnist']
#analysis.rd_curve(f, beta = True, name = 'fmnist_smalltrain')
#f = ['results/echo0init_smalltrain_binary_mnist', 'results/echo2add_smalltrain_binary_mnist', 'results/echo-2corrected_binary_mnist']#, 'results/alemi_test_vae_binary_mnist']
#analysis.rd_curve(f, beta = True, name = f[0].split('/')[-1], threshold = 0.000001)

#fn = 'alemi_echo_0.7.pickle'
#with open(os.path.join(os.getcwd(), f[1], fn), "rb") as pkl_data:
#   results = pickle.load(pkl_data)
#analysis.plot_loss(results, keys=['bce_recon_loss', 'mi_echo_reg_loss'], prefix = f[0]+'/')

# fn = 'alemi_echoAddBeta_0.5.pickle'
# with open(os.path.join(os.getcwd(), f[1], fn), "rb") as pkl_data:
#     results = pickle.load(pkl_data)
# analysis.plot_loss(results, keys=['bce_recon_loss', 'vae_reg_loss'], prefix = f[1]+'/')

# f = ['results/vae_beta', 'results/echo_add_beta']
# analysis.rd_curve(f, beta = True, name = 'echo_vs_vae_beta')

# f = ['results/ido_beta', 'results/echo_mult_beta']
# analysis.rd_curve(f, beta = True, name = 'echo_vs_ido_beta')

# f = ['results/vae_constraint', 'results/echo_add_constraint']
# analysis.rd_curve(f, beta = False, name = 'echo_vs_vae_constraint')

# f = ['results/ido_constraint', 'results/echo_mult_constraint']
# analysis.rd_curve(f, beta = False, name = 'echo_vs_ido_constraint')