import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np 
import h5py

from build import load_data
from sfm import SFM,LSTM
from train import train_lstm,test,train_sfm,test_sfm
from utils import plot_results,ToVariable,use_cuda
from option_parser import OptionParser

def main():
    parser = OptionParser()
    opt = parser.parse_args()

    filename = opt.filename
    step = opt.step
    epochs = opt.epochs
    
    x_train, y_train, x_val, y_val,x_test, y_test, gt_test, max_data, min_data = load_data(filename,step)

    opt.max_data = max_data
    opt.min_data = min_data
    opt.train_len = x_train.shape[1]
    opt.val_len = x_val.shape[1]-x_train.shape[1]
    opt.test_len = x_test.shape[1]-x_val.shape[1]

    lstm = LSTM(1,50,1)
    sfm = SFM(1,50,20,1)

    
    net = sfm
    train_function = train_sfm
    test_function = test_sfm

    if opt.net not in ["lstm","sfm"]:
        raise NameError("Undefined net!!")

    if opt.net == "lstm":
        net = lstm
        train_function = train_lstm
        test_function = test

    print("Using net: %s"%(opt.net))

    if opt.train:
        train_function(net,x_train,y_train,epochs=epochs)
    if opt.test:
        test_function(net,x_test,y_test,opt)

if __name__=='__main__':
    main()
