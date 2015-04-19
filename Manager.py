#-*- coding: UTF-8 -*-
'''
Created on 20150417

@author: wangjie
'''
import numpy as np

import PreProcessData
import PreRBMTrain
import Train
import ForwardPropagation
import Config




def read_batch_data():
    
    global X_train_batch
    global y_train_batch
    global X_CV_batch
    global y_CV_batch
    global X_test_batch
    global y_test_batch
    #merge train and CV
    global X_merge_train_batch
    global y_merge_train_batch


    batch_data = np.load(Config.FILE_BATCH_DATA_PATH)
    X_train_batch = batch_data['X_train_batch']
    y_train_batch = batch_data['y_train_batch']

    X_CV_batch = batch_data['X_CV_batch']
    y_CV_batch = batch_data['y_CV_batch']
    
    X_test_batch = batch_data['X_test_batch']
    y_test_batch = batch_data['y_test_batch']
    
    X_merge_train_batch = np.append(X_train_batch, X_CV_batch, axis = 0)
    y_merge_train_batch = np.append(y_train_batch, y_CV_batch, axis = 0)
    
    np.savez(Config.FILE_ZERO_RBM_PATH, train_data_output = X_test_batch)
    
if __name__ == '__main__':
    
    #PreProcessData.pre_process_data(Config.FILE_ORIGIN_ZIP_DATA_PATH, Config.FILE_BATCH_DATA_PATH)
    
    read_batch_data()
    
    PreRBMTrain.pre_rbm_train()

    #Train.train(X_merge_train_batch, y_merge_train_batch, Config.FILE_TRAIN_PATH)
    Train.train(X_test_batch, y_test_batch, Config.FILE_TRAIN_PATH)
    
    #print ForwardPropagation.forward(X_merge_train_batch, y_merge_train_batch)

    print ForwardPropagation.forward(X_test_batch, y_test_batch)
    
    
    
    