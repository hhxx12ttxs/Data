#-*- coding: UTF-8 -*-
'''
Created on 20150417

@author: wangjie
'''
import RBM
import Config

def pre_rbm_train():
    #pre train by rbm between two layers
    RBM.rbm(Config.FILE_ZERO_RBM_PATH, Config.HIDDEN_UNITS_NUM_FIRST_RBM, Config.FILE_FIRST_RBM_PATH)
    
    RBM.rbm(Config.FILE_FIRST_RBM_PATH, Config.HIDDEN_UNITS_NUM_SECOND_RBM, Config.FILE_SECOND_RBM_PATH)
        
    RBM.rbm(Config.FILE_SECOND_RBM_PATH, Config.HIDDEN_UNITS_NUM_THIRD_RBM, Config.FILE_THIRD_RBM_PATH)  
        
    RBM.rbm(Config.FILE_THIRD_RBM_PATH, Config.OUTPUT_UNITS_NUM_FORTH_RBM, Config.FILE_FORTH_RBM_PATH) 