#-*- coding: UTF-8 -*-
'''
Created on 20150415

@author: wangjie
'''

import numpy as np
from numpy.matlib import repmat

import Config
from Methods import sigmoid

def rbm(input_file_name, hidden_units_num, output_file_name):
    train_data = np.load(input_file_name)['train_data_output']
    
    batches_num, images_num_of_each_batch, image_pixes = train_data.shape
    
    train_data_output = np.zeros((batches_num, images_num_of_each_batch, hidden_units_num)) 
    
    visiable_weights = 0.1 * np.random.randn(image_pixes, hidden_units_num)
    visiable_weights_delta = np.zeros((image_pixes, hidden_units_num))
    
    visiable_biases = np.zeros((1, hidden_units_num))
    visiable_biases_delta = np.zeros((1, hidden_units_num))

    hidden_biases = np.zeros((1, image_pixes))
    hidden_biases_delta = np.zeros((1, image_pixes))
    #forward
    positive_probablity = np.zeros((images_num_of_each_batch, hidden_units_num))
    positive_product = np.zeros((image_pixes, hidden_units_num))
    #forward -> backward -> forward
    negative_probablity = np.zeros((images_num_of_each_batch, hidden_units_num))
    negative_product = np.zeros((image_pixes, hidden_units_num))
    
    for iteration in range(0, Config.RBM_ITERATION_NUM):
        for batch in range(0, batches_num):
            
            if iteration > 5:
                momentum = Config.FINAL_MOMENTUM
            else:
                momentum = Config.INITIAL_MOMENTUM
                
            data = train_data[batch, :, :]
            positive_probablity = sigmoid((np.dot(data, visiable_weights) + repmat(visiable_biases, images_num_of_each_batch, 1)))
            positive_probablity_states = positive_probablity > np.random.rand(images_num_of_each_batch, hidden_units_num)
            negative_data = sigmoid((np.dot(positive_probablity_states, np.transpose(visiable_weights)) + \
                                           repmat(hidden_biases, images_num_of_each_batch, 1)))
            negative_probablity = sigmoid(np.exp(np.dot(negative_data, visiable_weights) + 
                                              repmat(visiable_biases, images_num_of_each_batch, 1)))
            
            positive_product = np.dot(np.transpose(data), positive_probablity)
            negative_product = np.dot(np.transpose(negative_data), negative_probablity)
            positive_probablity_sum_of_rows = sum(positive_probablity)
            negative_probablity_sum_of_rows = sum(negative_probablity)
            data_sum_of_rows = sum(data)
            negative_data_sum_of_rows = sum(negative_data)

                
            visiable_weights_delta = momentum * visiable_weights_delta + (Config.EPSILON_WEIGHT / images_num_of_each_batch) *\
                                        (positive_product - negative_product) 
            visiable_biases_delta = momentum * visiable_biases_delta + (Config.EPSILON_VISIABLE / images_num_of_each_batch) * \
                                        (positive_probablity_sum_of_rows - negative_probablity_sum_of_rows)
            hidden_biases_delta = momentum * hidden_biases_delta + (Config.EPSILON_HIDDEN / images_num_of_each_batch) * \
                                        (data_sum_of_rows - negative_data_sum_of_rows)
            
            visiable_weights += visiable_weights_delta
            hidden_biases += hidden_biases_delta
            visiable_biases += visiable_biases_delta
            
            if iteration == Config.RBM_ITERATION_NUM - 1:
                train_data_output[batch, :, :] = positive_probablity

    np.savez(output_file_name, train_data_output = train_data_output, weights = visiable_weights, biases = visiable_biases)                                   

           
if __name__ == '__main__':
    pass