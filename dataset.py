import numpy as np
from abc import ABC, abstractmethod
from keras.datasets import mnist, cifar10
import os 
import gzip

class Dataset(ABC): 
    #def __init__(self, per_label = None):
    #    if per_label is not None
        
    @abstractmethod
    def get_data(self):
        pass

    def shrink_supervised(self, per_label):
        pass

    #def num_samples(self):

class DatasetWrap(Dataset):
    def __init__(self, x_train, x_val = None, y_train = None, y_val = None, x_test = None, y_test = None, dim1 = None, dim2 = None):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.dim = x_train.shape[-1]
        if dim1 is None:
            self.dim1 = np.sqrt(self.dim)
            if int(self.dim1) **2 == self.dim: # check integer sqrt
                self.dim2 = self.dim1
                self.dims = [self.dim1, self.dim2] 
            else:
                print("Please enter dim1, dim2 ")
        else:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dims = [self.dim1, self.dim2]  
    def get_data(self):
        return self.x_train, self.x_val, self.y_train, self.y_val

class MNIST(Dataset):
    def __init__(self, binary = False, val = 0, per_label = None):
        self.dims = [28,28]
        self.dim1 = 28
        self.dim2 = 28
        self.dim = 784
        self.binary = binary
        self.name = 'mnist' if not binary else 'binary_mnist'

        self.x_train, self.x_test, self.y_train, self.y_test = self.get_data()
        self.x_train = self.x_train[:(self.x_train.shape[0]-val), :]
        self.x_val = self.x_train[(self.x_train.shape[0]-val):, :]

        if per_label is not None:
            pass

    def get_data(self, onehot = False, path = None):
        if path is None and self.binary:
            path = './datasets/mnist/MNIST_binary'
        elif path is None:
            path = 'MNIST_data/mnist.npz'

        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])

        if not self.binary:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.
            x_train = x_train.reshape((len(x_train), self.dim))
            x_test = x_test.reshape((len(x_test), self.dim))
            # Clever one hot encoding trick
            if onehot:
                y_train = np.eye(10)[y_train]
                y_test = np.eye(10)[y_test]
            return x_train, x_test, y_train, y_test

        else:
            with open(os.path.join(path, 'binarized_mnist_train.amat')) as f:
                lines = f.readlines()
            train_data = lines_to_np_array(lines).astype('float32')
            with open(os.path.join(path, 'binarized_mnist_valid.amat')) as f:
                lines = f.readlines()
            validation_data = lines_to_np_array(lines).astype('float32')
            with open(os.path.join(path, 'binarized_mnist_test.amat')) as f:
                lines = f.readlines()
            test_data = lines_to_np_array(lines).astype('float32')


            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            print(train_data.shape, y_train.shape)
            return train_data, test_data, y_train, y_test
            #return train_data, validation_data, test_data

class fMNIST(Dataset):
    def __init__(self, binary = False, val = 0, per_label = None):
        self.dims = [28,28]
        self.dim1 = 28
        self.dim2 = 28
        self.dim = 784
        #unused
        #self.binary = binary
        self.name = 'fmnist'

        #self.x_train, self.x_test, self.y_train, self.y_test = self.get_data()
        self.x_train, self.y_train = self.get_data(kind='train')
        self.x_test, self.y_test = self.get_data(kind='test')
        self.x_train = self.x_train[:(self.x_train.shape[0]-val), :]
        self.x_val = self.x_train[(self.x_train.shape[0]-val):, :]

    def get_data(self, kind = 'train', path = None):
        if path is None:
            path = './datasets/mnist/fMNIST'

        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)
        
        images = images.astype('float32') / 255.
        images = images.reshape((len(images), self.dim))
        
        return images, labels



class DSprites(Dataset):
    def __init__(self):
        self.dims = [64,64]
        self.dim1 = 64
        self.dim2 = 64
        self.dim = 4096
        self.name = 'dsprites'
        self.x_train, _, self.labels, _ = self.get_data()
        self.x_val = None
        self.x_test = None

    def get_data(self):
        dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding = 'bytes') # ‘latin1’
        imgs = dataset_zip['imgs']
        imgs = imgs.reshape((imgs.shape[0], -1))
        latent_ground_truth = dataset_zip['latents_values']
        latent_for_classification = dataset_zip['latents_classes']
        metadata = dataset_zip['metadata']
        return imgs, latent_ground_truth, latent_for_classification, metadata

class Omniglot(Dataset):
    def __init__(self):
        self.dim1 = None
        self.dim2 = None
        self.dim = None
        self.name = 'omniglot'
        self.x_train = self.get_data()
        # TO DO: Create validation / test sets by resampling?
        self.x_val = None
        self.x_test = None 

    def get_data(self, per_char = 5, total_chars = 964, imsize = 105, seed = 0, file_dir = 'omniglot_train/'):
        imgs = np.random.randint(1, 20, (total_chars,per_char))
        x_train = np.zeros((total_chars*per_char, imsize**2))
        for i in range(total_chars):
            for j in range(per_char):
                k=imgs[i,j]
                fn = os.path.join(file_dir, str(i+1).zfill(4) + '_' + str(k).zfill(2) + '.png')
                x_train[i*per_char+j, :] = imread(fn, flatten = True).flatten()
        return x_train


class EMNIST(Dataset):
    def __init__(self):
        raise NotImplementedError
    def get_data(self):
        raise NotImplementedError

class CelebA(Dataset):
    def __init__(self):
        raise NotImplementedError
    def get_data(self):
        raise NotImplementedError

class Cifar10(Dataset):
    def __init__(self):
        raise NotImplementedError
    def get_data(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        return X_train, X_test, y_train, y_test
        #raise NotImplementedError