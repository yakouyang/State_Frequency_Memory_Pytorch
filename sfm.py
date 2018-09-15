import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F 
import numpy as np

class SFM(nn.Module):
    def __init__(self,input_size, hidden_size, freq_size,output_size):
        super(SFM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.freq_size = freq_size
        self.output_size = output_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size,output_size)

        self.omega = Variable(torch.tensor(2*np.pi*np.arange(1,self.freq_size+1)/self.freq_size).float())
        self.init_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_parameters(self):
        # gate weights
        # W -> x, U -> h_last
        self.i_W = Parameter(torch.randn(self.input_size,self.hidden_size))
        self.i_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.i_b = Parameter(torch.randn(self.hidden_size))
        
        self.g_W = Parameter(torch.randn(self.input_size,self.hidden_size))
        self.g_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.g_b = Parameter(torch.randn(self.hidden_size))

        self.o_W = Parameter(torch.randn(self.input_size,self.hidden_size))
        self.o_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.o_V = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.o_b = Parameter(torch.randn(self.hidden_size))

        self.fre_W = Parameter(torch.randn(self.input_size,self.freq_size)) # belong R^K
        self.fre_U = Parameter(torch.randn(self.hidden_size,self.freq_size))
        self.fre_b = Parameter(torch.randn(self.freq_size))

        self.ste_W = Parameter(torch.randn(self.input_size,self.hidden_size))
        self.ste_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.ste_b = Parameter(torch.randn(self.hidden_size))

        self.a_U = Parameter(torch.randn(self.freq_size,1))
        self.a_b = Parameter(torch.randn(self.hidden_size))

    def forward(self,input,h,c,re_s,im_s,time):
        i_t = self.sigmoid(torch.matmul(input,self.i_W)+torch.matmul(h,self.i_U)+self.i_b)
        c_hat_t = self.sigmoid(torch.matmul(input,self.g_W)+torch.matmul(h,self.g_U)+self.g_b)

        f_ste = self.sigmoid(torch.matmul(input,self.ste_W)+torch.matmul(h,self.ste_U)+self.ste_b) # belong R^D
        f_fre = self.sigmoid(torch.matmul(input,self.fre_W)+torch.matmul(h,self.fre_U)+self.fre_b) # belong R^K
        f_t = torch.matmul(f_ste.view(-1,self.hidden_size,1),f_fre.view(-1,1,self.freq_size))
        
        re_s = torch.mul(f_t,re_s) + torch.matmul(torch.mul(i_t,c_hat_t).transpose(1,2),torch.cos(torch.mul(self.omega,time)).unsqueeze(0))
        im_s = torch.mul(f_t,im_s) + torch.matmul(torch.mul(i_t,c_hat_t).transpose(1,2),torch.sin(torch.mul(self.omega,time)).unsqueeze(0))

        a_t = torch.sqrt(re_s**2+im_s**2)
        c_t = self.tanh(torch.matmul(a_t,self.a_U).transpose(1,2)+self.a_b)

        o_t = self.sigmoid(torch.matmul(input,self.o_W)+torch.matmul(h,self.o_U)+torch.matmul(c_t,self.o_V)+self.o_b)
        h_t = torch.mul(o_t,self.tanh(c_t))
        output = self.output(h_t)
        output = output.view(-1,1)
        time += 1
        return output,h_t,c_t,re_s.squeeze(0),im_s.squeeze(0),time

    def init_state(self):
        h = Variable(torch.zeros(1, self.hidden_size))
        c = Variable(torch.zeros(1, self.hidden_size))
        re_s = Variable(torch.zeros(self.hidden_size, self.freq_size))
        im_s = Variable(torch.zeros(self.hidden_size, self.freq_size))
        time = Variable(torch.ones(1))
        return h,c,re_s,im_s,time

class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(LSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size,output_size)

        self.set_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def set_parameters(self):
        # gate weights
        # W -> x, U -> h_last
        self.f_W = Parameter(torch.randn(self.input_size,self.hidden_size))
        self.f_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.f_b = Parameter(torch.randn(self.hidden_size))

        self.i_W = Parameter(torch.randn(self.input_size,self.hidden_size))
        self.i_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.i_b = Parameter(torch.randn(self.hidden_size))
        
        self.g_W = Parameter(torch.randn(self.input_size,self.hidden_size))
        self.g_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.g_b = Parameter(torch.randn(self.hidden_size))

        self.o_W = Parameter(torch.randn(self.input_size,self.hidden_size))
        self.o_U = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.o_V = Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.o_b = Parameter(torch.randn(self.hidden_size))

    def forward(self,input,h,c):
        i_t = self.sigmoid(torch.matmul(input,self.i_W)+torch.matmul(h,self.i_U)+self.i_b)
        f_t = self.sigmoid(torch.matmul(input,self.f_W)+torch.matmul(h,self.f_U)+self.f_b)
        c_hat_t = self.sigmoid(torch.matmul(input,self.g_W)+torch.matmul(h,self.g_U)+self.g_b)

        c_t = torch.mul(i_t,c_hat_t) + torch.mul(f_t,c)
        o_t = self.sigmoid(torch.matmul(input,self.o_W)+torch.matmul(h,self.o_U)+torch.matmul(c,self.o_V)+self.o_b)
        h_t = torch.mul(o_t,self.tanh(c_t))
        output = self.output(h_t)
        output = output.view(-1,1)
        return output,h_t,c_t

    def init_state(self):
        h = Variable(torch.zeros(1, self.hidden_size))
        c = Variable(torch.zeros(1, self.hidden_size))
        return h,c