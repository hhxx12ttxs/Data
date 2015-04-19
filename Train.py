#-*- coding: UTF-8 -*-
'''
Created on 201504015

@author: wangjie
'''

import numpy as np
from numpy.matlib import repmat
from numpy import transpose

import Config
from Methods import sigmoid, sigmoidInv


def train(data, target, file_name):
    #load weights and biases of rbm 
    '''
    first_rbm = np.load(Config.FILE_FIRST_RBM_PATH)
    first_weights = first_rbm['weights']
    first_biases = first_rbm['biases']
    
    second_rbm = np.load(Config.FILE_SECOND_RBM_PATH)
    second_weights = second_rbm['weights']
    second_biases = second_rbm['biases']
    
    third_rbm = np.load(Config.FILE_THIRD_RBM_PATH)
    third_weights = third_rbm['weights']
    third_biases = third_rbm['biases']
    
    forth_rbm = np.load(Config.FILE_FORTH_RBM_PATH)
    forth_weights = forth_rbm['weights']
    forth_biases = forth_rbm['biases']
    '''
        
    batches_num, images_num_of_each_batch, image_pixes = data.shape
    
    first_weights = 0.1 * np.random.randn(image_pixes, Config.HIDDEN_UNITS_NUM_FIRST_RBM)
    first_biases = np.zeros((1, Config.HIDDEN_UNITS_NUM_FIRST_RBM))
    
    second_weights = 0.1 * np.random.randn(Config.HIDDEN_UNITS_NUM_FIRST_RBM, Config.HIDDEN_UNITS_NUM_SECOND_RBM)
    second_biases = np.zeros((1, Config.HIDDEN_UNITS_NUM_SECOND_RBM))
    
    third_weights = 0.1 * np.random.randn(Config.HIDDEN_UNITS_NUM_SECOND_RBM, Config.HIDDEN_UNITS_NUM_THIRD_RBM)
    third_biases = np.zeros((1, Config.HIDDEN_UNITS_NUM_THIRD_RBM))
    
    forth_weights = 0.1 * np.random.randn(Config.HIDDEN_UNITS_NUM_THIRD_RBM, Config.OUTPUT_UNITS_NUM_FORTH_RBM)
    forth_biases = np.zeros((1, Config.OUTPUT_UNITS_NUM_FORTH_RBM))
    
    for iteration in range(0, Config.FORWARD_BACKWARD_ITERATION_NUM):
        classify_images_right_num = 0
        for batch in range(0, batches_num):
            
            batch_data = data[batch, :, :]

            second_x = np.dot(batch_data, first_weights) + repmat(first_biases, images_num_of_each_batch, 1)
            second_probablity = sigmoid(second_x)
            
            
            third_x = np.dot(second_probablity, second_weights) + repmat(second_biases, images_num_of_each_batch, 1)
            third_probablity = sigmoid(third_x)
            
            forth_x = np.dot(third_probablity, third_weights) + repmat(third_biases, images_num_of_each_batch, 1)
            forth_probablity = sigmoid(forth_x)
            
            fifth_x = np.dot(forth_probablity, forth_weights) + repmat(forth_biases, images_num_of_each_batch, 1)
            fifth_probablity = sigmoid(fifth_x)

            diff = target[batch, :, :].argmax(axis = 1) - fifth_probablity.argmax(axis = 1)
            classify_images_right_num += sum(diff == 0)
            '''
            #average sum of squares error term
            Jcost = (0.5 / images_num_of_each_batch) * sum(sum((fifth_probablity - target[batch, :, :]) * (fifth_probablity - target[batch, :, :])))
            #weight decay term
            Jweight = 0.5 * ((sum(sum(first_weights * first_weights))) + (sum(sum(second_weights * second_weights))) +\
                               (sum(sum(third_weights * third_weights))) + (sum(sum(forth_weights * forth_weights))))
            #used to verify
            cost = Jcost + Config.LAMBDA * Jweight
            '''
            fifth_max = np.max(fifth_probablity, axis = 1)
            fifth_max.shape = (images_num_of_each_batch, 1)
            fifth_probablity = fifth_probablity >= repmat(fifth_max, 1, Config.DEMICAL)
            
            fifth_error = (fifth_probablity - target[batch, :, :]) * sigmoidInv(fifth_x)
            
            forth_error = np.dot(fifth_error, transpose(forth_weights)) * sigmoidInv(forth_x)
 
            third_error = np.dot(forth_error, transpose(third_weights)) * sigmoidInv(third_x)

            second_error = np.dot(third_error, transpose(second_weights)) * sigmoidInv(second_x)
            '''
            #sparse auto encoder
            forth_rho = (1. / images_num_of_each_batch) * forth_probablity.sum(axis = 0)
            forth_sterm = Config.BETA * ( - Config.SPARSITY_PARAM / forth_rho + (1 - Config.SPARSITY_PARAM) / (1 - forth_rho))
            forth_error = (np.dot(fifth_error, transpose(forth_weights)) + repmat(forth_sterm, images_num_of_each_batch, 1)) * sigmoidInv(forth_x)

            third_rho = (1. / images_num_of_each_batch) * third_probablity.sum(axis = 0)
            third_sterm = Config.BETA * ( - Config.SPARSITY_PARAM / third_rho + (1 - Config.SPARSITY_PARAM) / (1 - third_rho))
            third_error = (np.dot(forth_error, transpose(third_weights)) + repmat(third_sterm, images_num_of_each_batch, 1)) * sigmoidInv(third_x)
            
            second_rho = (1. / images_num_of_each_batch) * second_probablity.sum(axis = 0)
            second_sterm = Config.BETA * ( - Config.SPARSITY_PARAM / second_rho + (1 - Config.SPARSITY_PARAM) / (1 - second_rho))
            second_error = (np.dot(third_error, transpose(second_weights)) + repmat(second_sterm, images_num_of_each_batch, 1)) * sigmoidInv(second_x)
            '''
            first_weights_derivative = (1. / images_num_of_each_batch) * (np.dot(transpose(batch_data), second_error)) + Config.LAMBDA * first_weights
            first_weights -= Config.ALPHA * first_weights_derivative
            first_biases_derivative = (1. / images_num_of_each_batch) * second_error.sum(axis = 0)
            first_biases -= Config.ALPHA * first_biases_derivative

            second_weights_derivative = (1. / images_num_of_each_batch) * (np.dot(transpose(second_x), third_error)) + Config.LAMBDA * second_weights
            second_weights -= Config.ALPHA * second_weights_derivative
            second_biases_derivative = (1. / images_num_of_each_batch) * third_error.sum(axis = 0)
            second_biases -= Config.ALPHA * second_biases_derivative
            
            third_weights_derivative = (1. / images_num_of_each_batch) * (np.dot(transpose(third_x), forth_error)) + Config.LAMBDA * third_weights
            third_weights -= Config.ALPHA * third_weights_derivative
            third_biases_derivative = (1. / images_num_of_each_batch) * forth_error.sum(axis = 0)
            third_biases -= Config.ALPHA * third_biases_derivative
            
            forth_weights_derivative = (1. / images_num_of_each_batch) * (np.dot(transpose(forth_x), fifth_error)) + Config.LAMBDA * forth_weights
            forth_weights -= Config.ALPHA * forth_weights_derivative
            forth_biases_derivative = (1. / images_num_of_each_batch) * fifth_error.sum(axis = 0)
            forth_biases -= Config.ALPHA * forth_biases_derivative
    np.savez(file_name, 
             first_weights = first_weights, first_biases = first_biases, 
             second_weights = second_weights, second_biases = second_biases, 
             third_weights = third_weights, third_biases = third_biases, 
             forth_weights = forth_weights, forth_biases = forth_biases)         
     
if __name__ == '__main__':
    pass