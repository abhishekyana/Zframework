import numpy as np

def one_hot(S1,word_mode=False):
    """

    This one_hot model is used to one_hot encode the data:
    input: S1:Sting or Integer to be one_hot encoded, mode_word: for word or letter mode.
    returns list of [un,oh] -> un: header file for unique values, oh: one_hot array(2D)

    Example:>>>one_hot(123456)
                list->[array([[1, 2, 3, 4, 5, 6]]),
                array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])]

    """
    S=S1
    if type(S1)==int:
        S=list(map(int,str(S1)))
    if type(S1)==str:
        S=list(S1)
        if word_mode==True:
            S=S1.split(' ')
    a=np.array(S)
    a=a.reshape(a.size,1)
    un=np.unique(a)
    un1=un.reshape(1,un.size)
    oh=(a==un1)*1
    return [un,oh]
