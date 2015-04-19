#-*- coding: UTF-8 -*-
'''
Created on 20150416

@author: wangjie

'''
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidInv(x):
    return sigmoid(x) * (1 - sigmoid(x))