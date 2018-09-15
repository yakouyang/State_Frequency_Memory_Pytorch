import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

use_cuda = torch.cuda.is_available()

def recover(data,length=50):
    for i in range(1,length):
        data[i,:,:] = data[0,:,:]
    return data

def plot_results(predicted_data, true_data):
    # use in train.py 
    # plot evaluate result
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def ToVariable(x):
    # use in train.py 
    # change from numpy.array to torch.variable   
    tmp = torch.FloatTensor(x)
    if use_cuda:
        return Variable(tmp).cuda()
    else:
        return Variable(tmp)

def shuffle_data(X,Y):
    data_train = [(x,y) for x,y in zip(X, Y)]
    np.random.shuffle(data_train)
    X = np.array([x for x,y in data_train])
    Y = np.array([y for x,y in data_train])
    return X,Y

