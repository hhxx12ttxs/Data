#-*- coding: UTF-8 -*-
'''
Created on 20150417

@author: wangjie
'''
import numpy as np
import Config
from Methods import sigmoid
from numpy.matlib import repmat

def forward(data, target):
    #load weights and biases of train
    train_output = np.load(Config.FILE_TRAIN_PATH)
    first_weights = train_output['first_weights']
    first_biases = train_output['first_biases']
    
    second_weights = train_output['second_weights']
    second_biases = train_output['second_biases']
    
    third_weights = train_output['third_weights']
    third_biases = train_output['third_biases']
    
    forth_weights = train_output['forth_weights']
    forth_biases = train_output['forth_biases']
        
    batches_num, images_num_of_each_batch, image_pixes = data.shape
    
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

    return classify_images_right_num * 1.0 / (batches_num * images_num_of_each_batch)