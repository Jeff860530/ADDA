from keras.datasets import mnist
import scipy.io as sio
import urllib.request
import shutil
import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

NUM_CLASSES = 10

def get_mnist():
   
   (train_x, train_y), (test_x, test_y) = mnist.load_data()
   train_x = np.pad(train_x,((0,0), (2,2), (2,2)),'constant')
   test_x = np.pad(test_x,((0,0), (2,2), (2,2)),'constant')
   train_x = np.stack([train_x]*3, axis=3)
   test_x = np.stack([test_x]*3, axis=3)
   
   return (train_x, train_y), (test_x, test_y) 

def get_svhn():
    
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    
    if not os.path.exists(os.path.join('datasets', 'svhn_train.mat')):
        print ('Downloading SVHN training set!')
        with urllib.request.urlopen('http://ufldl.stanford.edu/housenumbers/train_32x32.mat') as response, open(os.path.join('datasets', 'svhn_train.mat'), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    
    if not os.path.exists(os.path.join('datasets', 'svhn_test.mat')):
        print ('Downloading SVHN test set!')
        with urllib.request.urlopen('http://ufldl.stanford.edu/housenumbers/test_32x32.mat') as response, open(os.path.join('datasets', 'svhn_test.mat'), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    
    train = sio.loadmat(os.path.join('datasets', 'svhn_train.mat'))
    test = sio.loadmat(os.path.join('datasets', 'svhn_test.mat'))
    
    train_y = train['y'].flatten()
    test_y = test['y'].flatten()
    train_y[train_y==10]=0
    test_y[test_y==10]=0
    
    train_x = np.transpose(train['X'], (3,0,1,2))
    test_x = np.transpose(test['X'], (3,0,1,2)) 
    
    '''
    Tx = np.zeros(train_x.shape[:-1])
    tx = np.zeros(test_x.shape[:-1])
    
    # Convert to grayscale
    Tx = 0.3*train_x[:,:,:,0] + 0.59*train_x[:,:,:,1] + 0.11*train_x[:,:,:,2]
    tx = 0.3*test_x[:,:,:,0] + 0.59*test_x[:,:,:,1] + 0.11*test_x[:,:,:,2]
    
    Tx = np.reshape(Tx, Tx.shape+(1,))
    tx = np.reshape(tx, tx.shape+(1,))
    '''
    
    return (train_x, train_y), (test_x, test_y) 

def get_dataset(dataset='mnist'):
    
    if dataset=='mnist':
        (train_x, train_y), (test_x, test_y) = get_mnist()
    elif dataset=='svhn':
        (train_x, train_y), (test_x, test_y) = get_svhn()
    
    train_y = np_utils.to_categorical(train_y, NUM_CLASSES)
    test_y = np_utils.to_categorical(test_y, NUM_CLASSES)
    
    return (train_x, train_y), (test_x, test_y)

if __name__=='__main__':

    (train_x, train_y), (test_x, test_y) = get_dataset('svhn')
    print (train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    (train_x, train_y), (test_x, test_y) = get_dataset('mnist')
    print (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
