""" Run MNIST experiments with Echo
"""
import numpy as np
import sys, os
sys.path.append('..')
import echo
import IPython
import tensorflow as tf
from vis_routines import *
from sklearn.metrics import accuracy_score

# Load data
x_train, x_test, y_train, y_test = load_data('cifar10')  # or 'cifar10', 'cifar100'

# Options
noise = 'correlated'
noise_type = 'multiplicative'  # multiplicative or additive noise
d_max = 100  # Maximum echo number... bigger is THEORETICALLY better, but more computationally intensive
batch_size = 1024  # batch_size >= d_max + 1
epochs = 100  # Checkpoints saved every 1000 epochs
learning_rate = 1e-3
debug = ('-d' in sys.argv)
beta = 1.
arch = {'type': 'conv',  # 'fc' for fully connected or 'conv' for convolutions
        'layers': [256, 128, 64, 2],
        'activation': tf.nn.softplus}  # a TF function. Ignored for multiplicative which always uses softplus

# Run it
out = echo.EchoSup(verbose=True, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, debug=debug,
                   categorical=True, beta=beta,
                   noise=noise, noise_type=noise_type, d_max=d_max,  # controlling the noise
                   architecture_args=arch)
output_dir = out.log_dir
save_script(output_dir, os.path.realpath(__file__), os.path.realpath(echo.__file__))  # Save script before fitting
out.fit(x_train, y_train, val_data=x_test[-batch_size:], val_labels=y_test[-batch_size:])
# out = echo.load('/path/to/model_999.ckpt', (28, 28))  # Specify checkpoint prefix and data shape to load

if '-i' in sys.argv:
    IPython.embed()

# Visualize results
y_test_predict = out.predict(x_test)
test_score = accuracy_score(y_test, y_test_predict)
y_train_predict = out.predict(x_train)
train_score = accuracy_score(y_train, y_train_predict)
result_string = 'train: {:.4f}, test: {:.4f}'.format(train_score, test_score)
print(result_string)
with open('{}/accuracy.txt'.format(output_dir), 'w') as f:
    f.write(result_string)

np.set_printoptions(suppress=True, precision=3, linewidth=220)
mis = out.measures['I(Z_j;X)']
print("The MI(Z_j;X) are bounded by these capacities:")
print(mis)

if arch['layers'][-1] == 2:  # Visualizations for 2-d space
    z_test = out.transform(x_test)
    if out.multiplicative:
        z_test = np.log(z_test.clip(1e-10))  # Should be nearly log-normal, more convenient to inspect in normal space
    scatter(z_test, 0, 1, y_test, output='{}/'.format(output_dir))

if '-i' in sys.argv:
    IPython.embed()