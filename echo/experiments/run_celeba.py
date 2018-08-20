""" Run MNIST experiments with Echo
"""
import numpy as np
import scipy.misc
import sys, os
sys.path.append('..')
import echo
import IPython
import tensorflow as tf
from scipy.linalg import sqrtm, pinvh, eigh
from vis_routines import *
from sklearn.covariance import MinCovDet

# Load data
x_train, x_test, _, _ = load_data('celeba')

# Options
noise = 'correlated'
noise_type = 'multiplicative'  # multiplicative or additive noise
binary_inputs = False  # use binary cross entropy loss for reconstruction if True
d_max = 50  # Maximum echo number... bigger is THEORETICALLY better, but more computationally intensive
batch_size = 1024  # batch_size >= d_max + 1
epochs = 10000  # Checkpoints saved every 1000 epochs
learning_rate = 1e-3
debug = ('-d' in sys.argv)
arch = {'type': 'fc',  # 'fc' for fully connected or 'conv' for convolutions
        'layers': [256, 128, 64, 2],
        'activation': tf.nn.softplus}  # a TF function. Ignored for multiplicative which always uses softplus

# Run it
out = echo.Echo(verbose=True, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, debug=debug,
                noise=noise, noise_type=noise_type, d_max=d_max,  # controlling the noise
                binary_inputs=binary_inputs, architecture_args=arch)
output_dir = out.log_dir
save_script(output_dir, os.path.realpath(__file__), os.path.realpath(echo.__file__))  # Save script before fitting
out.fit(x_test, val_data=x_test[-batch_size:])
# out = echo.load('/home/gregv/tmp/tensorflow/0140/model_4999.ckpt', (218, 178, 3))  # Specify checkpoint prefix and data shape to load
# output_dir = out.log_dir

# Visualize results
# 1. RECONSTRUCTION
z_test = out.transform(x_test[:500])
x_reconstruction = out.predict(z_test)
if out.binary_inputs:
    x_reconstruction = 1. / (1. + np.exp(- x_reconstruction))
scipy.misc.imsave('{}/reconstruction_big.png'.format(output_dir),
             np.vstack([np.hstack(echo.load_data(x_test[:40])), np.hstack(x_reconstruction[:40])]))

if out.multiplicative:
    z_test = np.log(z_test.clip(1e-10))  # Should be nearly log-normal, more convenient to inspect in normal space
mu = np.mean(z_test, axis=0)
cov = MinCovDet().fit(z_test).covariance_  # np.cov(z_test.T)
w, v = eigh(cov)
T = np.dot(np.sqrt(w.clip(1e-10)) * v, v.T)
np.set_printoptions(suppress=True, precision=3, linewidth=220)
print("The latent space has mean:")
print(mu)
print("and covariance:")
print(cov)
mis = out.measures['I(Z_j;X)']
print("The MI(Z_j;X) are bounded by these capacities:")
print(mis)

if '-i' in sys.argv:
    IPython.embed()

if arch['layers'][-1] == 2:  # Visualizations for 2-d space
    # 2. MANIFOLD
    manifold(out, mu, T, output='{}/rotated_'.format(output_dir))
    manifold(out, mu, np.identity(2), output='{}/'.format(output_dir))
    pylab.clf()
else:  # Visualizations for n-d space
    d1, d2 = np.argsort(-mis)[:2]
    for i in range(10):
        manifold_middle(z_test[i], out, d1, d2, mu, T, output='{}/{}'.format(output_dir, i), w=16)

if '-i' in sys.argv:
    IPython.embed()