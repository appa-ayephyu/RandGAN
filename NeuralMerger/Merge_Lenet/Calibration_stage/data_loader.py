import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/fashion', one_hot=False)
from scipy.misc import imresize
import warnings
warnings.filterwarnings("ignore")

def load_sound20_data(config):

    '''
    Sound20 (32*32)
    Traing  data:16636
    Val     data:3249   (we do not use)
    Testing data:3727
    '''
    if config.use_testing == False:
        train_num       = 16636
        train_X         = np.load("./data/train_X.npy")
        #train_X         = np.load("./data/train_X.npy", allow_pickle=True, encoding = 'bytes')
        train_Y         = np.load("./data/train_Y.npy")
        #train_Y         = np.load("./data/train_Y.npy", allow_pickle=True, encoding = 'bytes')
        
        if (config.use_1000_training):
            calibration_num = 1000
        else:
            calibration_num = int(train_num*config.data_ratio)
        np.random.seed(config.random_seed)
        random_shuffle  = np.random.choice(train_num,calibration_num,replace = False)
        train_X_cal     = np.take(train_X,random_shuffle,axis=0)
        train_Y_cal     = np.take(train_Y,random_shuffle,axis=0)
        train_X_cal     = train_X_cal.reshape([calibration_num,32,32,1])

    else:
        train_X         = np.load("./data/test_X.npy")
        train_Y_cal     = np.load("./data/test_Y.npy")
        train_X_cal     = train_X.reshape([3727,32,32,1])

    test_X         = np.load("./data/test_X.npy")
    test_Y         = np.load("./data/test_Y.npy")


    
    test_X          = test_X.reshape([3727,32,32,1])
    print(train_X_cal.shape)
    return train_X_cal,train_Y_cal,test_X,test_Y

def load_fasion_data(config):
    '''
    MNIST Fashion (28*28) -> (32*32)
    Traing  data:55000
    Val     data:5000 (we do not use)
    Testing data:10000
    '''

    test_X_     = mnist.test.images #(10000,784)
    test_Y      = mnist.test.labels #(10000,1)
    test_X_     = test_X_.reshape(10000,28,28)
    test_X      = np.array(np.random.normal(size=[10000,32,32]),dtype=np.float32)
    for i in range(10000):
        test_X[i] = imresize(test_X_[i],(32,32))
    test_X          = test_X.reshape([10000,32,32,1])


    if config.use_testing == False:
        train_X_        = mnist.train.images #(55000,784)
        train_Y         = mnist.train.labels #(55000,1)
        train_X_        = train_X_.reshape(55000,28,28)
        train_X         = np.array(np.random.normal(size=[55000,32,32]))
        for i in range(55000):
            train_X[i]  = imresize(train_X_[i],(32,32))

        train_num       = 55000
        if (config.use_1000_training):
            calibration_num = 1000
        else:
            calibration_num = int(train_num*config.data_ratio)



        np.random.seed(config.random_seed)
        random_shuffle  = np.random.choice(train_num,calibration_num,replace = False)
        train_X_cal     = np.take(train_X,random_shuffle,axis=0)
        train_Y_cal     = np.take(train_Y,random_shuffle,axis=0)

        train_X_cal     = train_X_cal.reshape([calibration_num,32,32,1])
    else:
        train_X_cal     = test_X
        train_Y_cal     = test_Y

    #plt.imshow(test_X[0].reshape(32,32))
    #plt.show()
    print(train_X_cal.shape)
    return train_X_cal,train_Y_cal,test_X,test_Y


