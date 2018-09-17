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

f = ['results/infovae_beta_binary_mnist']#, 'results/alemi_test_vae_binary_mnist']
analysis.rd_curve(f, beta = True, name = 'echo_vae_mnist')

f = ['results/echo_add_alemi_.001_binary_mnist']#, 'results/alemi_test_vae_binary_mnist']
analysis.rd_curve(f, beta = True, name = 'echo_alemi_fmnist')

f = ['results/alemi_vae_beta_fmnist']#, 'results/alemi_test_vae_binary_mnist']
analysis.rd_curve(f, beta = True, name = 'vae_alemi_fmnist')

fn = 'echo_add_alemi_.001_0.8.pickle'
with open(os.path.join(os.getcwd(), f[0], fn), "rb") as pkl_data:
    results = pickle.load(pkl_data)
analysis.plot_loss(results, keys=['bce_recon_loss', 'mi_echo_reg_loss'], prefix = f[0]+'/')

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