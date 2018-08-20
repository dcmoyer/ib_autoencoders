""" Run MNIST experiments with Echo
"""
import numpy as np
import sys, os
sys.path.append('..')
import echo
import IPython
import tensorflow as tf
from scipy.linalg import sqrtm, pinvh, eigh
from vis_routines import *
from sklearn.decomposition import PCA, FastICA
from sklearn.covariance import MinCovDet

# Load data
x_train, x_test, y_train, y_test = load_data('mnist')

# Options
noise = 'correlated'
noise_type = 'additive' #'multiplicative'  # multiplicative or additive noise
binary_inputs = True  # use binary cross entropy loss for reconstruction if True
d_max = 50  # Maximum echo number... bigger is THEORETICALLY better, but more computationally intensive
batch_size = 1024  # batch_size >= d_max + 1
epochs = 100  # Checkpoints saved every 1000 epochs
learning_rate = 1e-3
debug = ('-d' in sys.argv)
arch = {'type': 'fc',  # 'fc' for fully connected or 'conv' for convolutions
        'layers': [128, 64, 8],
        'activation': tf.nn.softplus}  # a TF function. Ignored for multiplicative which always uses softplus

# Run it
out = echo.Echo(verbose=True, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, debug=debug,
                noise=noise, noise_type=noise_type, d_max=d_max,  # controlling the noise
                binary_inputs=binary_inputs, architecture_args=arch)
output_dir = out.log_dir
save_script(output_dir, os.path.realpath(__file__), os.path.realpath(echo.__file__))  # Save script before fitting
out.fit(x_train, val_data=x_test[-batch_size:])
# out = echo.load('/path/to/model_999.ckpt', (28, 28))  # Specify checkpoint prefix and data shape to load

# Visualize results
# 1. RECONSTRUCTION
z_test = out.transform(x_test)
x_reconstruction = out.predict(z_test)
if out.binary_inputs:
    x_reconstruction = 1. / (1. + np.exp(- x_reconstruction))
all_dig = [0,1,2,3,4,7,8,11,18,61]  # One of each digit
pylab.imsave('{}/reconstruction.png'.format(output_dir),
             np.vstack([np.hstack(x_test[all_dig]), np.hstack(x_reconstruction[all_dig])]), vmin=0, vmax=1, cmap='gray')
pylab.clf()
pylab.imsave('{}/reconstruction_big.png'.format(output_dir),
             np.vstack([np.hstack(x_test[:40]), np.hstack(x_reconstruction[:40])]), vmin=0, vmax=1, cmap='gray')
pylab.clf()

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

    # 3. Scatter plot of digits
    scatter(z_test, 0, 1, y_test, output='{}/'.format(output_dir))
    z_test2 = PCA(whiten=True).fit_transform(z_test)  # Rotated basis
    scatter(z_test2, 0, 1, y_test, output='{}/rotated_'.format(output_dir))
else:  # Visualizations for n-d space
    try:
        d1, d2 = np.argsort(-mis)[:2]
    except:
        d1, d2 = 0, 1
    scatter(z_test, d1, d2, y_test, output='{}/top2_'.format(output_dir))
    for i in all_dig:
        manifold_middle(z_test[i], out, d1, d2, mu, T, output='{}/{}'.format(output_dir, mnist.test.labels[i]))

output_dir += '/comparisons'
os.makedirs(output_dir)
knn_results = "Method, \t\t kNN mean accuracy"
knn_results = "Raw data,\t\t0.9677\n"
z_train = out.transform(x_train)
z_test = out.transform(x_test)
knn_score = knn_metric(z_train, y_train, z_test, y_test)
knn_results += "Echo,\t\t{:0.4f}\n".format(knn_score)
rotate = PCA(whiten=True).fit(out.transform(x_train))
knn_score = knn_metric(rotate.transform(z_train), y_train,
                       rotate.transform(z_test), y_test)
knn_results += "Echo rotated,\t\t{:0.4f}\n".format(knn_score)

# Comparisons with ICA, PCA
d = arch['layers'][-1]
if d < 16:
    for method_name, method in [('PCA', PCA), ('ICA', FastICA)]:
        out = method(n_components=d)
        z_train = out.fit_transform(x_train.reshape((-1, 784)))
        z_test = out.transform(x_test.reshape((-1, 784)))
        knn_score = knn_metric(z_train, y_train, z_test, y_test)
        knn_results += "{},\t\t{:0.4f}\n".format(method_name, knn_score)
        if d == 2:
            scatter(z_test, 0, 1, y_test, output='{}/{}_'.format(output_dir, method_name))

print(knn_results)
with open('{}/knn.txt'.format(output_dir), 'w') as f:
    f.write(knn_results)

if '-i' in sys.argv:
    IPython.embed()