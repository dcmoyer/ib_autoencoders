""" Visualization routines to use for experiments.
"""
import sys, os, shutil
import numpy as np
import pylab
from sklearn.neighbors import KNeighborsClassifier
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
import glob
import pickle


def load_data(dataset='mnist'):
    if dataset == 'cifar10':
        extract = lambda q: q['data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).astype(np.float32)
        tdict = pickle.load(open('cifar-10-batches-py/test_batch', 'rb'))
        x_test = extract(tdict)
        y_test = np.array(tdict['labels'])
        dicts = [pickle.load(open(f, 'rb')) for f in glob.glob('cifar-10-batches-py/data_batch_*')]
        x_trains = map(extract, dicts)
        x_train = np.concatenate(x_trains)
        y_train = np.array(sum([d['labels'] for d in dicts], []))
        return x_train, x_test, y_train, y_test
    elif dataset == 'cifar100':
        extract = lambda q: q['data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).astype(np.float32)
        tdict = pickle.load(open('cifar-100-python/test', 'rb'))
        x_test = extract(tdict)
        y_test = np.array(tdict['fine_labels'])
        tdict = pickle.load(open('cifar-100-python/train', 'rb'))
        x_train = extract(tdict)
        y_train = np.array(tdict['fine_labels'])
        return x_train, x_test, y_train, y_test
    elif dataset == 'mnist':
        mnist = input_data.read_data_sets("MNIST_data/")
        x_train = mnist.train.images.reshape((-1, 28, 28))
        y_train = mnist.train.labels
        x_test = mnist.test.images.reshape((-1, 28, 28))
        y_test = mnist.test.labels
        return x_train, x_test, y_train, y_test
    elif dataset == 'celeba':
        # TODO: get train/val/test partition, get labels
        data = glob.glob(os.path.join('img_align_celeba', '*.jpg'))
        x_train = data[:100000]
        x_test = data[100000:110000]
        return x_train, x_test, None, None


def save_script(output_dir, this_script, this_code_version):
    # Save experiment files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if this_script[-1] == 'c':
        this_script = this_script[:-1]
    if this_code_version[-1] == 'c':
        this_code_version = this_code_version[:-1]
    shutil.copyfile(this_script, output_dir + '/script_used.py')
    shutil.copyfile(this_code_version, output_dir + '/echo_version_used.py')


def manifold(model, mu=0, T=np.identity(2), output='manifold'):
    """For a 2-d space, plot the manifold assuming standard normal Gaussian.
        Given mu, covariance T, can also plot a non-standard Gaussian."""
    ys = np.arange(-2., 2., 0.1)
    mat = np.empty((28 * len(ys), 0))
    for y1 in ys:
        this_col = np.empty((0, 28))
        for y2 in ys:
            this_y = np.dot(T, np.array([y1, y2])) + mu
            if model.multiplicative:
                this_y = np.exp(this_y)
            x = model.predict(np.array([this_y]))[0]
            if model.binary_inputs:
                x = 1./(1.+np.exp(-x))
            this_col = np.vstack([x, this_col])
        mat = np.hstack([mat, this_col])
        imsave('{}manifold.png'.format(output), mat)
    #pylab.imsave('{}manifold.png'.format(output), mat, vmin=0, vmax=1, cmap='gray')
    #pylab.clf()


def manifold_middle(y_middle, model, d1, d2, mu=np.zeros(2), T=np.identity(2), output='manifold', w=40):
    """Plot a 2-d subspace around a given center latent point."""
    # mu = mu[[d1, d2]]
    # T = T[np.ix_([d1, d2], [d1, d2])]
    sig1, sig2 = np.sqrt(T[d1, d1]), np.sqrt(T[d2, d2])
    ys = np.arange(-2., 2.1, 4./w)
    im_grid = []
    for y1 in ys:
        im_grid.append([])
        for y2 in ys:
            this_y = np.copy(y_middle)
            this_y[d1] += y1 * sig1
            this_y[d2] += y2 * sig2
            if model.multiplicative:
                this_y = np.exp(this_y)
            x = model.predict(np.array([this_y]))[0]
            if model.binary_inputs:
                x = 1./(1.+np.exp(-x))
            im_grid[-1].append(x)
    n = len(ys)
    height, width = x.shape[:2]
    if len(x.shape) > 2:
        mat = np.array(im_grid).swapaxes(1, 2).reshape(height * n, width * n, -1)
    else:
        mat = np.array(im_grid).swapaxes(1, 2).reshape(height * n, width * n)
    # Make a border around the middle test point
    # mat[20 * 28, 20*28:21*28] = np.nan
    # mat[21 * 28, 20*28:21*28] = np.nan
    # mat[20*28:21*28, 20 * 28] = np.nan
    # mat[20*28:21*28, 21 * 28] = np.nan
    imsave('{}manifold.png'.format(output), mat)
    #pylab.imsave('{}manifold.png'.format(output), mat, vmin=0, vmax=1, cmap='gray')
    #pylab.clf()


def scatter(z, d1, d2, test_labels, output=''):
    """A scatter plot of latent factors on some 2-d subspace, with points colored according to test labels."""
    tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i in np.unique(test_labels):
        inds = (test_labels == i)
        pylab.scatter(z[inds, d1], z[inds, d2], marker='.', color=tab[i], alpha=0.5, edgecolor='', label=i)
    pylab.legend(loc=2)
    pylab.xlabel("$Z_{}$".format(d1))
    pylab.ylabel("$Z_{}$".format(d2))
    mu = np.mean(z, axis=0)
    std = np.std(z, axis=0)
    pylab.xlim(mu[d1] - 3 * std[d1], mu[d1] + 3 * std[d1])
    pylab.ylim(mu[d2] - 3 * std[d2], mu[d2] + 3 * std[d2])
    pylab.title('Latent space')
    pylab.savefig('{}latent_scatter.png'.format(output))
    pylab.clf()


def knn_metric(x_train, y_train, x_test, y_test):
    cls = KNeighborsClassifier(n_neighbors=1)
    cls.fit(x_train.reshape((x_train.shape[0], -1)), y_train)
    return cls.score(x_test.reshape((x_test.shape[0], -1)), y_test)
