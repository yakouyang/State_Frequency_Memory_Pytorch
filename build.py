import numpy as np
#Load data from data file, and split the data into training, validation and test set

def load_data(filename, step):
    #load data from the data file
    day = step
    data = np.load(filename)
    data = data[:, :]
    gt_test = data[:,day:]
    #data normalization
    max_data = np.max(data, axis = 1)
    min_data = np.min(data, axis = 1)
    max_data = np.reshape(max_data, (max_data.shape[0],1))
    min_data = np.reshape(min_data, (min_data.shape[0],1))
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)
    #dataset split
    train_split = round(0.8 * data.shape[1])
    val_split = round(0.9 * data.shape[1])
    
    x_train = data[:,:train_split]
    y_train = data[:,day:train_split+day]
    x_val = data[:,:val_split]
    y_val = data[:,day:val_split+day]
    x_test = data[:,:-day]
    y_test = data[:,day:]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    return [x_train, y_train, x_val, y_val, x_test, y_test, gt_test, max_data, min_data]

def load_roll_data(filename, seq_len, step,choose=0):
    #load data from the data file
    data = np.load(filename)
    data = data[:, :]
    day = step
    #data normalization
    max_data = np.max(data, axis = 1)
    min_data = np.min(data, axis = 1)
    max_data = np.reshape(max_data, (max_data.shape[0],1))
    min_data = np.reshape(min_data, (min_data.shape[0],1))
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)
    #dataset split
    train_split = round(0.8 * data.shape[1])
    val_split = round(0.9 * data.shape[1])

    x_data = []
    y_data = []
    i = 0
    print("len_data",data.shape[1])
    
    while((i+seq_len+step) <= data.shape[1]) :
        x_temp = data[choose,i:(i+seq_len)]
        y_temp = data[choose,(i+seq_len):(i+seq_len+step)]
        
        x_data.append(x_temp)
        y_data.append(y_temp)
        i += 1

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print("x_data",x_data.shape)
    print("y_data",y_data.shape)
    print(train_split)
    print(val_split)

    x_train = x_data[:train_split]
    y_train = y_data[:train_split]
    x_val = x_data[train_split:val_split]
    y_val = y_data[train_split:val_split]
    x_test = x_data[val_split:]
    y_test = y_data[val_split:]
    #print(x_train.shape)
    
    x_train = np.reshape(x_train, (x_train.shape[0], seq_len, 1))
    x_val = np.reshape(x_val, (x_val.shape[0], seq_len, 1))
    x_test = np.reshape(x_test, (x_test.shape[0],seq_len , 1))

    y_train = np.reshape(y_train, (y_train.shape[0], step,1))
    y_val = np.reshape(y_val, (y_val.shape[0],step , 1))
    y_test = np.reshape(y_test, (y_test.shape[0],step ,1))
    
    return x_train, y_train, x_val, y_val, x_test, y_test, max_data, min_data