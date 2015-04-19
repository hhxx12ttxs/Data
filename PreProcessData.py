#-*- coding: UTF-8 -*-
'''
Created on 20150415

@author: wangjie
'''
import gzip, cPickle
import numpy as np
import os
import Config

def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()   
    return train_set[0], train_set[1], valid_set[0], valid_set[1], test_set[0], test_set[1]  

def convert_data_into_batches(data, depth, row, col):
    batch_data = np.zeros((depth, row, col))
    for k in range(0, depth):
        begin_row = k * row
        end_row = (k + 1) * row
        batch_data[k, :, :] = data[begin_row : end_row , :]
    return batch_data 

def encode_demical(data, depth, row, col): 
    batch_data = np.zeros((depth, row, col))
    for k in range(0, depth):
        for i in range(0, row):
            batch_data[k, i, data[k * row + i]] = 1
    return batch_data 

def pre_process_data(data_path, file_name):
    Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data(data_path)
    Xtrain = np.array(Xtrain)
    XCV    = np.array(XCV)
    Xtest  = np.array(Xtest)
    X_train_batch = convert_data_into_batches(Xtrain, Config.TRAIN_BATCHES_NUM, 
                                              Config.IMAGE_NUM_OF_EACH_BATCH, Config.IMAGE_PIXES)
    y_train_batch = encode_demical(ytrain, Config.TRAIN_BATCHES_NUM, 
                                              Config.IMAGE_NUM_OF_EACH_BATCH, Config.DEMICAL)
    X_CV_batch = convert_data_into_batches(XCV, Config.VALIDATE_BATCHES_NUM,
                                              Config.IMAGE_NUM_OF_EACH_BATCH, Config.IMAGE_PIXES)
    y_CV_batch = encode_demical(yCV, Config.VALIDATE_BATCHES_NUM, 
                                              Config.IMAGE_NUM_OF_EACH_BATCH, Config.DEMICAL)
    X_test_batch = convert_data_into_batches(Xtest, Config.TEST_BATCHES_NUM, 
                                              Config.IMAGE_NUM_OF_EACH_BATCH, Config.IMAGE_PIXES)
    y_test_batch = encode_demical(ytest, Config.TEST_BATCHES_NUM, 
                                              Config.IMAGE_NUM_OF_EACH_BATCH, Config.DEMICAL)
    np.savez(file_name,
             X_train_batch = X_train_batch, y_train_batch = y_train_batch,
             X_CV_batch = X_CV_batch, y_CV_batch = y_CV_batch, 
             X_test_batch = X_test_batch, y_test_batch = y_test_batch)
    
if __name__ == '__main__':
    pre_process_data()
    '''
    Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data(Config.FILE_ORIGIN_ZIP_DATA_PATH)
    Xtrain = np.array(Xtrain)
    XCV    = np.array(XCV)
    Xtest  = np.array(Xtest)
    X_CV_batch = convert_data_into_batches(XCV, Config.VALIDATE_BATCHES_NUM, 
                                              Config.IMAGE_NUM_OF_EACH_BATCH, Config.IMAGE_PIXES)
    y_CV_batch = encode_demical(yCV, Config.VALIDATE_BATCHES_NUM, 
                                              Config.IMAGE_NUM_OF_EACH_BATCH, Config.DEMICAL)
    '''
    

    