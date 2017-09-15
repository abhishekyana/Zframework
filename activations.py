import numpy as np

def activate(type,z):

    if type=='relu':
        return np.maximum(0,z)
    if type=='sigmoid':
        return 1/(1+np.exp(-z))

def activate_prime(type,a):

    if type=='relu':
        return a>=0
    if type=='sigmoid':
        return a*(1-a)
