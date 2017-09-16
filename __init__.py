import numpy as np
import matplotlib.pyplot as plt
import pickle as p
from .activations import *
from .losses import *

def load_data(filename,train=False, test=False):
    """
    Used to get the pickle loaded data file into the program
    returns : Rdata: list of data in order Xtrain,Ytrain,Xtest,Ytest if needed.
    """
    data=p.load(open(filename,'rb'))
    Rdata=[]
    if train==True:
        Rdata.append(data["X_train"])
        Rdata.append(data["Y_train"])
    if test==True:
        Rdata.append(data["X_test"])
        Rdata.append(data["Y_test"])
    return Rdata


class model():

    def __init__(self,Xtrain,Ytrain,layerdims,activations=[None],loss_type='logistic'):
        self.Xtrain=Xtrain
        self.Ytrain=Ytrain
        self.loss_type=loss_type
        self.n_x=Xtrain.shape[0]
        self.n_y=Ytrain.shape[0]
        layerdims=list(layerdims)
        activations.insert(0,None)
        self.activations=activations
        layerdims.insert(0,self.n_x)
        layerdims.append(self.n_y)
        self.layerdims=layerdims
        self.nolayers=len(layerdims)
        assert(len(activations)==len(layerdims))
        self.initialize_parameters()
        print(f"Initialized model with {self.nolayers}(input->output) with dimensions {self.layerdims}")

    # def load_data(self,filename):
    #     self.data=p.load(open(filename,'rb'))
    #     self.Xtrain,self.Ytrain=self.data["X_train"],self.data["Y_train"]

    def initialize_parameters(self):
        W=[None]*(self.nolayers)
        b=[None]*(self.nolayers)

        for l in range(1,self.nolayers):
            W[l]=np.random.randn(self.layerdims[l],self.layerdims[l-1])*0.01
            b[l]=np.zeros((self.layerdims[l],1))

        self.parameters={"W":W,"b":b}
        self.W=W;self.b=b
        return self.parameters

    def load_parameters(self,parameters_filename):
        self.parameters=p.load(open(parameters_filename,'rb'))
        self.W=self.parameters["W"]
        self.b=self.parameters["b"]

    def forward_propagate(self):
        self.Z=[None]*len(self.W)
        self.A=[None]*(len(self.W)-1)
        self.A.insert(0,self.X)
        for l in range(1,self.nolayers):
            self.Z[l]=np.dot(self.W[l],self.A[l-1])+self.b[l]
            self.A[l]=activate(self.activations[l],self.Z[l])
        self.A_out=self.A[-1]
        self.cache={"Z":self.Z,"A":self.A}
        #return A_out,cache


    def make_batch(self,test=False):
        vals=np.random.choice(self.Xtrain.shape[1],self.batch_size)
        self.X,self.Y=self.Xtrain[:,vals],self.Ytrain[:,vals]
        self.m=self.X.shape[1]
        if test==True:
            vals=np.random.choice(self.Xtest.shape[1],self.batch_size)
            self.X,self.Y=self.Xtest[:,vals],self.Ytest[:,vals]
            self.m=self.X.shape[1]


#####Newly added without proper modifications########
    def compute_cost(self):
        """
        This computes the cost to be reduced using the logisic error function
        inputs: A_out, Y, loss_type {'logistic'}
        outputs: unnormalized cost. should be divided with the total examples
        """
        logs=loss(self.loss_type,self.Y,self.A_out)
        cost=np.sum(logs)/self.m
        self.cost=np.squeeze(cost)

    def backprop(self):
        #Y,cache,parameters,grads
        #acts=cache["acts"]
        #W=self.parameters["W"]
        #b=parameters["b"]
        #m=Y.shape[1]
        dZ=[None]*len(self.Z)
        dA=[None]*len(self.A)
        dW=[None]*len(self.W) #self.grads["dW"]
        db=[None]*len(self.b) #self.grads["db"]
        dA[-1]=loss_prime(self.loss_type,self.Y,self.A_out) #for logistic loss
        for layer in range((len(self.Z)-1),0,-1):
            dZ[layer]=np.multiply(dA[layer],activate_prime(self.activations[layer],self.A[layer]))
            dW[layer]=np.dot(dZ[layer],self.A[layer-1].T)/self.m; db[layer]=np.sum(dZ[layer],axis=1,keepdims=True)/self.m
            dA[layer-1]=np.dot(self.W[layer].T,dZ[layer])
        self.grads={"dW":dW,"db":db}
        self.dW=dW
        self.db=db
        #return grads
    def update_parameters(self):
        for layer in range(1,len(self.W)):
            self.W[layer]-=self.dW[layer]*self.learning_rate
            self.b[layer]-=self.db[layer]*self.learning_rate

    def predict(self,threshold=0.5):
        self.predicted=self.A_out>=threshold

    def train(self,learning_rate=0.01,epoches=10,batch_size=64):
        """
        This function is to be implemented..
        """
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        for epoch in range(1,epoches+1):
            self.make_batch()
            self.forward_propagate()
            self.compute_cost()
            print(self.cost)
            self.backprop()
            self.update_parameters()

    def test(self,xx,yy):
        self.X=xx
        self.Y=yy
        self.forward_propagate()
        #self.compute_cost()
        #print(self.A_out)
        return self.A_out

    def evaluate(self,Xtest,Ytest):
        self.Xtest=Xtest
        self.Ytest=Ytest
        self.make_batch(test=True)
        self.forward_propagate()
        self.compute_cost()
        self.predict()
        accuracy=self.predicted==self.Y
        print(f"Accuracy : {(accuracy.sum()/accuracy.size)*100}%")



"""
import Z
model=Z.model(784,10,2,[100,100],['relu','relu','sigmoid'])
prms=model.initialize_parameters()
model.load_data('fmnistdata.d')
model.make_batch()
model.forward_propagate()


The mini batches model is trained according to the random.choice function and the list of selected indexes are passed over full training  data to get the minibatches to the function
"""
