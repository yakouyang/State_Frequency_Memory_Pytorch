import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import shuffle_data,plot_results,ToVariable,use_cuda
#torch.backends.cudnn.deterministic = True

def train_lstm(model,x_train,y_train,epochs=10):
    optimizer = optim.RMSprop(model.parameters())
    criterion = nn.MSELoss()

    X,Y = shuffle_data(x_train,y_train)

    x_len = X.shape[1]
    X = ToVariable(X)
    Y = ToVariable(Y)
    for epoch in range(0,epochs):
        h,c = model.init_state()
        for step in range(0,x_len):
            x = X[:,step,:].view(-1,1,1)
            y = Y[:,step,:]
                
            out_put,h,c = model(x,h,c)
            h = h.data
            c = c.data

            optimizer.zero_grad()
            loss = criterion(out_put, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            print('Epoch: ', epoch+1, '| step: ', step+1, '| Loss: ',loss.detach())

    torch.save(model, 'model/model.pkl')

def test(model,x_test,y_test,opt):
    model = torch.load('model/model.pkl')
    model.eval()
    pred_dat = []

    h,c = model.init_state()    
    seq_len = x_test.shape[1]

    for i in range(0,seq_len):
        x = ToVariable(x_test[:,i,:])
        x = x.view(-1,1,1)
        pre_out,h,c =  model(x,h,c)
        h = h.data
        c = c.data
        if use_cuda:
            pred_dat.append(pre_out.data.cpu().numpy())
        else:
            pred_dat.append(pre_out.data.numpy())
        
    pred_dat=np.array(pred_dat)

    pred_dat = pred_dat.transpose(1,0,2)
    pred_dat = (pred_dat[:,:, 0] * (opt.max_data - opt.min_data) + (opt.max_data + opt.min_data))/2
    y_test = (y_test[:,:, 0] * (opt.max_data - opt.min_data) + (opt.max_data + opt.min_data))/2

    error = np.sum((pred_dat[:,-opt.test_len:] - y_test[:,-opt.test_len:])**2) / (opt.test_len* pred_dat.shape[0])
    print('The mean square error is: %f' % error)

    plot_results(pred_dat[0,-opt.test_len:],y_test[0,-opt.test_len:])
    
def train_sfm(model,x_train,y_train,epochs=10):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    X,Y = shuffle_data(x_train,y_train)

    x_len = X.shape[1]
    X = ToVariable(X)
    Y = ToVariable(Y)

    for epoch in range(0,epochs):
        h,c,re_s,im_s,time = model.init_state()
        for step in range(0,x_len):

            x = X[:,step,:]
            y = Y[:,step,:]

            x = x.unsqueeze(1)
                
            output,h,c,re_s,im_s,time = model(x,h,c,re_s,im_s,time)
            h = h.data
            c = c.data
            re_s = re_s.data
            im_s = im_s.data
            time = time.data

            loss = criterion(output.squeeze(0), y)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            print('Epoch: ', epoch+1, '| step: ', step+1, '| Loss: ',loss.detach())

    torch.save(model, 'model/model2.pkl')

def test_sfm(model,x_test,y_test,opt):
    model = torch.load('model/model2.pkl')
    model.eval()
    pred_dat = []

    h,c,re_s,im_s,time = model.init_state()
    seq_len = x_test.shape[1]

    for i in range(0,seq_len):
        x = ToVariable(x_test[:,i,:])
        x = x.view(-1,1,1)
        pre_out,h,c,re_s,im_s,time =  model(x,h,c,re_s,im_s,time)
        h = h.data
        c = c.data
        re_s = re_s.data
        im_s = im_s.data
        time = time.data
        if use_cuda:
            pred_dat.append(pre_out.data.cpu().numpy())
        else:
            pred_dat.append(pre_out.data.numpy())

    pred_dat=np.array(pred_dat)
    pred_dat = pred_dat.transpose(1,0,2)
    pred_dat = (pred_dat[:,:, 0] * (opt.max_data - opt.min_data) + (opt.max_data + opt.min_data))/2
    y_test = (y_test[:,:, 0] * (opt.max_data - opt.min_data) + (opt.max_data + opt.min_data))/2

    error = np.sum((pred_dat[:,-opt.test_len:] - y_test[:,-opt.test_len:])**2) / (opt.test_len* pred_dat.shape[0])
    print('The mean square error is: %f' % error)

    plot_results(pred_dat[0,-opt.test_len:],y_test[0,-opt.test_len:])