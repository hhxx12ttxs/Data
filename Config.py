#-*- coding: UTF-8 -*-
'''
Created on 20150415

@author: wangjie
'''

TRAIN_BATCHES_NUM = 500
VALIDATE_BATCHES_NUM = 100
TEST_BATCHES_NUM = 100
IMAGE_NUM_OF_EACH_BATCH = 100
IMAGE_PIXES = 784
DEMICAL = 10

#learning rate for weights in rbm
EPSILON_WEIGHT = 0.1
#learning rate for visiable units biases in rbm
EPSILON_VISIABLE = 0.1
#learning rate for hidden units biases in rbm
EPSILON_HIDDEN = 0.1
WEIGHT_COST = 0.0002
INITIAL_MOMENTUM = 0.5
FINAL_MOMENTUM = 0.9

'''
#the argumets of sparse autoencoder which are not used
#desired average activation of the hidden units
SPARSITY_PARAM = 0.01
#weight of sparsity penalty term 
BETA = 0.005
'''
#weight decay parameter
LAMBDA = 0.0005

#learing rate
ALPHA = 0.02

#iteration num of rbm
RBM_ITERATION_NUM = 20

FORWARD_BACKWARD_ITERATION_NUM = 20

HIDDEN_UNITS_NUM_FIRST_RBM = 500
HIDDEN_UNITS_NUM_SECOND_RBM = 500
HIDDEN_UNITS_NUM_THIRD_RBM = 500
OUTPUT_UNITS_NUM_FORTH_RBM = 10

#':' just same as '\'
FILE_ORIGIN_ZIP_DATA_PATH = "input\mnist.pkl.gz"
FILE_BATCH_DATA_PATH = "output\\origin_batch_data.npz"
FILE_ZERO_RBM_PATH = "output\\zero_rbm.npz"
FILE_FIRST_RBM_PATH = "output\\first_rbm.npz"
FILE_SECOND_RBM_PATH = "output\\second_rbm.npz"
FILE_THIRD_RBM_PATH = "output\\third_rbm.npz"
FILE_FORTH_RBM_PATH = "output\\forth_rbm.npz"
FILE_TRAIN_PATH = "output\\train.npz"


