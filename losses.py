import numpy as np

def loss(type,Y,A_out):
    if type=='logistic':
        return -(np.multiply(Y,np.log(A_out))+np.multiply((1-Y),np.log(1-A_out)))
    if type=='mean_square_error':
        return np.square(A_out-Y)

def loss_prime(type,Y,A_out):
    if type=='logistic':
        return -(np.divide(Y,A_out)-np.divide((1-Y),(1-A_out)))
    if type=='mean_square_error':
        return A_out-Y
