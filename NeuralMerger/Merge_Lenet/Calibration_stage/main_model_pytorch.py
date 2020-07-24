# note: I combined the original main.py and model.py together under this file for the pytorch version

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from data_loader import load_sound20_data, load_fasion_data
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
from weight_loader import load_weight_data
from config import *
import time

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = setting()

# get weights and other informations
newpath = r'./bin_file'
M_codebook, M1_w, M1_b, M1_index, M1_ch_order, M2_w, M2_b, M2_index, M2_ch_order,mean_adjust = load_weight_data(config)

start = time.time()


def embedding_idx(index,k_num):

    for i in range(index.shape[0]): 
        index[i] = index[i] + k_num*i 
    return index

# define parameters and settings
config = config
codebook  = M_codebook

print ("Calibration Iter   : %d"%config.max_iter)
if config.use_1000_training: 
    print ("Calibration Data   : 1000")
else:
    print ("Calibration Data_r : %.2f"%config.data_ratio)
print ("Learning Rate      : %.4f"%config.lr_rate) # learning rate = 0.0001
print ("Batch Size         : %d"%config.batch_size) # batch size = 64 
        
w1     = M1_w # [(5, 5, 1, 32), (5, 5, 32, 64), (4096, 1024), (1024, 20) # network 1's original weights
b1     = M1_b # [(32,), (64,), (1024,), (20,)]  network1's original bias
i1     = M1_index # [(1,800), (1,1600), (512,1024)] segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to)
align1 = M1_ch_order # [(1,), (32,), (4096,)] 0 to chn in - 1
i1_len = len(i1)
for i in range(i1_len):
    i1[i]     =   embedding_idx(i1[i],codebook[i].shape[1])
    i1[i]    =   torch.from_numpy(i1[i]).long()

c  = []
for i in range(3): 
    c.append(codebook[i].reshape(codebook[i].shape[0]*codebook[i].shape[1],codebook[i].shape[2]))
    c[i] = torch.from_numpy(c[i])
        
w2     = M2_w # [(5, 5, 1, 32), (5, 5, 32, 64), (4096, 1024), (1024, 10) # network 2's original weights
b2     = M2_b # [(32,), (64,), (1024,), (10,)]  network2's original bias
i2     = M2_index # [(1,800), (1,1600), (512,1024)] segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to)
align2 = M2_ch_order # [(1,), (32,), (4096,)] 0 to chn in - 1
i2_len = len(i2)        
for i in range(i2_len):
    i2[i]    =   embedding_idx(i2[i],codebook[i].shape[1])   
    i2[i]    =   torch.from_numpy(i2[i]).long()

lr_rate  = config.lr_rate

# obtain the new weights of a network from the codebook
def get_weight_from_codebook(c, index, w3):
    
    w0= []
    for i in range(index[0].shape[0]):
        w0.append(torch.index_select(c[0], 0, index[0][i]))
    w0 = torch.stack(w0, dim=0)
    w0 = w0.permute(0,2,1)
    w0 = w0.reshape(index[0].shape[0]*int(c[0].shape[1]),5,5,32)
    w0 = w0.permute(1,2,0,3)

    w1= []
    for i in range(index[1].shape[0]):
        w1.append(torch.index_select(c[1], 0, index[1][i]))
    w1 = torch.stack(w1, dim=0)
    w1 = w1.permute(0,2,1)
    w1 = w1.reshape(index[1].shape[0]*int(c[1].shape[1]),5,5,64)
    w1 = w1.permute(1,2,0,3)

    w2=  []
    for i in range(index[2].shape[0]):
        w2.append(torch.index_select(c[2], 0, index[2][i]))
    w2 = torch.stack(w2, dim=0)
    w2 = w2.permute(0,2,1)
    w2 = w2.reshape(index[2].shape[0]*int(c[2].shape[1]),1024)

    w3= w3
    
    return np.array([w0.numpy(),w1.numpy(),w2.numpy(),w3])

# compute the difference between well trained weights and new weights
def get_weight_loss(new_w, old_w):
    new_w = torch.tensor(new_w)
    old_w = torch.from_numpy(old_w)
    return torch.mean(torch.abs(new_w-old_w))


class Well_trained_M1(nn.Module):
    
    def __init__(self, w, b):
        super(Well_trained_M1, self).__init__()        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)       
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool2 =nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(4096, 1024)
        
        self.fc2 = nn.Linear(1024, 20)  
        

        # assign weights and bias
        for i in range(len(w)):
            if i == 2 or i == 3:
                w[i] = w[i].transpose([1,0])
            elif i == 0 or i == 1:
                w[i] = w[i].transpose([3,2,0,1])
        
        c1 = torch.from_numpy(w[0])
        c1 = torch.tensor(c1, requires_grad=False)

        b1 = torch.from_numpy(b[0])
        b1 = torch.tensor(b1, requires_grad=False)

        c2 = torch.from_numpy(w[1])
        c2 = torch.tensor(c2, requires_grad=False)

        b2 = torch.from_numpy(b[1])
        b2 = torch.tensor(b2, requires_grad=False)

        f1 = torch.from_numpy(w[2])
        f1 = torch.tensor(f1, requires_grad=False)

        b3 = torch.from_numpy(b[2])
        b3 = torch.tensor(b3, requires_grad=False)

        f2 = torch.from_numpy(w[3])
        f2 = torch.tensor(f2, requires_grad=False)

        b4 = torch.from_numpy(b[3])
        b4 = torch.tensor(b4, requires_grad=False)
        
        self.conv1.weight = torch.nn.parameter.Parameter(c1)
        self.conv1.bias = torch.nn.parameter.Parameter(b1)
        self.conv2.weight = torch.nn.parameter.Parameter(c2)
        self.conv2.bias = torch.nn.parameter.Parameter(b2)
        self.fc1.weight = torch.nn.parameter.Parameter(f1)
        self.fc1.bias = torch.nn.parameter.Parameter(b3)
        self.fc2.weight = torch.nn.parameter.Parameter(f2)
        self.fc2.bias = torch.nn.parameter.Parameter(b4)
        
        for i in range(len(w)):
            if i == 2 or i == 3:
                w[i] = w[i].transpose([1,0])
            elif i == 0 or i == 1:
                w[i] = w[i].transpose([2,3,1,0])

    def forward(self, x):
        c1_out = self.maxpool1(F.relu(self.conv1(x)))  
        c2_out = self.maxpool2(F.relu(self.conv2(c1_out))).permute(0,2,3,1)
        fc1_out = F.relu(self.fc1(c2_out.reshape(c2_out.size(0), -1)))
        fc2_out = self.fc2(fc1_out)
        return [c1_out.permute(0,2,3,1), c2_out, fc1_out, fc2_out]
    
class Well_trained_M2(nn.Module): 

    def __init__(self, M2_w, M2_b):
        super(Well_trained_M2, self).__init__()        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)      
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool2 =nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(4096, 1024)
        
        self.fc2 = nn.Linear(1024, 10)  
        
        # assign weights and bias
        for i in range(len(M2_w)):
            if i == 2 or i == 3:
                M2_w[i] = M2_w[i].transpose([1,0])
            elif i == 0 or i == 1:
                M2_w[i] = M2_w[i].transpose([3,2,0,1])
        
        c1 = torch.from_numpy(M2_w[0])
        c1 = torch.tensor(c1, requires_grad=False, dtype=torch.double)

        b1 = torch.from_numpy(M2_b[0])
        b1 = torch.tensor(b1, requires_grad=False, dtype=torch.double)

        c2 = torch.from_numpy(M2_w[1])
        c2 = torch.tensor(c2, requires_grad=False, dtype=torch.double)

        b2 = torch.from_numpy(M2_b[1])
        b2 = torch.tensor(b2, requires_grad=False, dtype=torch.double)

        f1 = torch.from_numpy(M2_w[2])
        f1 = torch.tensor(f1, requires_grad=False, dtype=torch.double)

        b3 = torch.from_numpy(M2_b[2])
        b3 = torch.tensor(b3, requires_grad=False, dtype=torch.double)

        f2 = torch.from_numpy(M2_w[3])
        f2 = torch.tensor(f2, requires_grad=False, dtype=torch.double)

        b4 = torch.from_numpy(M2_b[3])
        b4 = torch.tensor(b4, requires_grad=False, dtype=torch.double)
        
        self.conv1.weight = torch.nn.parameter.Parameter(c1)
        self.conv1.bias = torch.nn.parameter.Parameter(b1)
        self.conv2.weight = torch.nn.parameter.Parameter(c2)
        self.conv2.bias = torch.nn.parameter.Parameter(b2)
        self.fc1.weight = torch.nn.parameter.Parameter(f1)
        self.fc1.bias = torch.nn.parameter.Parameter(b3)
        self.fc2.weight = torch.nn.parameter.Parameter(f2)
        self.fc2.bias = torch.nn.parameter.Parameter(b4)

        for i in range(len(M2_w)):
            if i == 2 or i == 3:
                M2_w[i] = M2_w[i].transpose([1,0])
            elif i == 0 or i == 1:
                M2_w[i] = M2_w[i].transpose([2,3,1,0])

    def forward(self, x):
        c1_out = self.maxpool1(F.relu(self.conv1(x)))   
        c2_out = self.maxpool2(F.relu(self.conv2(c1_out))).permute(0,2,3,1)
        fc1_out = F.relu(self.fc1(c2_out.reshape(c2_out.size(0), -1)))
        fc2_out = self.fc2(fc1_out)
        return [c1_out.permute(0,2,3,1), c2_out, fc1_out, fc2_out]

class PQ_M1(nn.Module):
    def __init__(self, M1_w, M1_b):
        super(PQ_M1, self).__init__()        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)       
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool2 =nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(4096, 1024)
        
        self.fc2 = nn.Linear(1024, 20)  
        
        # assign weights and bias
        for i in range(len(M1_w)):
            if i == 2 or i == 3:
                M1_w[i] = M1_w[i].transpose([1,0])
            elif i == 0 or i == 1:
                M1_w[i] = M1_w[i].transpose([3,2,0,1])
        
        c1 = torch.from_numpy(M1_w[0])
        c1 = torch.tensor(c1, requires_grad=True)

        b1 = torch.from_numpy(M1_b[0])
        b1 = torch.tensor(b1, requires_grad=True)

        c2 = torch.from_numpy(M1_w[1])
        c2 = torch.tensor(c2, requires_grad=True)

        b2 = torch.from_numpy(M1_b[1])
        b2 = torch.tensor(b2, requires_grad=True)

        f1 = torch.from_numpy(M1_w[2])
        f1 = torch.tensor(f1, requires_grad=True)

        b3 = torch.from_numpy(M1_b[2])
        b3 = torch.tensor(b3, requires_grad=True)

        f2 = torch.from_numpy(M1_w[3])
        f2 = torch.tensor(f2, requires_grad=True)

        b4 = torch.from_numpy(M1_b[3])
        b4 = torch.tensor(b4, requires_grad=True)
        
        self.conv1.weight = torch.nn.parameter.Parameter(c1)
        self.conv1.bias = torch.nn.parameter.Parameter(b1)
        self.conv2.weight = torch.nn.parameter.Parameter(c2)
        self.conv2.bias = torch.nn.parameter.Parameter(b2)
        self.fc1.weight = torch.nn.parameter.Parameter(f1)
        self.fc1.bias = torch.nn.parameter.Parameter(b3)
        self.fc2.weight = torch.nn.parameter.Parameter(f2)
        self.fc2.bias = torch.nn.parameter.Parameter(b4)

        for i in range(len(M1_w)):
            if i == 2 or i == 3:
                M1_w[i] = M1_w[i].transpose([1,0])
            elif i == 0 or i == 1:
                M1_w[i] = M1_w[i].transpose([2,3,1,0])

    def forward(self, x):
        c1_out = self.maxpool1(F.relu(self.conv1(x)))
        c2_out = self.maxpool2(F.relu(self.conv2(c1_out))).permute(0,2,3,1)
        fc1_out = F.relu(self.fc1(c2_out.reshape(c2_out.size(0), -1)))
        fc2_out = F.relu(self.fc2(fc1_out))
        return [c1_out.permute(0,2,3,1), c2_out, fc1_out, fc2_out]

class PQ_M2(nn.Module): 
    def __init__(self, M1_w, M1_b):
        super(PQ_M2, self).__init__()        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)       
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool2 =nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(4096, 1024)
        
        self.fc2 = nn.Linear(1024, 10)  
        
        # assign weights and bias
        for i in range(len(M1_w)):
            if i == 2 or i == 3:
                M1_w[i] = M1_w[i].transpose([1,0])
            elif i == 0 or i == 1:
                M1_w[i] = M1_w[i].transpose([3,2,0,1])
        
        c1 = torch.from_numpy(M1_w[0])
        c1 = torch.tensor(c1, requires_grad=True, dtype=torch.double)

        b1 = torch.from_numpy(M1_b[0])
        b1 = torch.tensor(b1, requires_grad=True, dtype=torch.double)

        c2 = torch.from_numpy(M1_w[1])
        c2 = torch.tensor(c2, requires_grad=True, dtype=torch.double)

        b2 = torch.from_numpy(M1_b[1])
        b2 = torch.tensor(b2, requires_grad=True, dtype=torch.double)

        f1 = torch.from_numpy(M1_w[2])
        f1 = torch.tensor(f1, requires_grad=True, dtype=torch.double)

        b3 = torch.from_numpy(M1_b[2])
        b3 = torch.tensor(b3, requires_grad=True, dtype=torch.double)

        f2 = torch.from_numpy(M1_w[3])
        f2 = torch.tensor(f2, requires_grad=True, dtype=torch.double)

        b4 = torch.from_numpy(M1_b[3])
        b4 = torch.tensor(b4, requires_grad=True, dtype=torch.double)
        
        self.conv1.weight = torch.nn.parameter.Parameter(c1)
        self.conv1.bias = torch.nn.parameter.Parameter(b1)
        self.conv2.weight = torch.nn.parameter.Parameter(c2)
        self.conv2.bias = torch.nn.parameter.Parameter(b2)
        self.fc1.weight = torch.nn.parameter.Parameter(f1)
        self.fc1.bias = torch.nn.parameter.Parameter(b3)
        self.fc2.weight = torch.nn.parameter.Parameter(f2)
        self.fc2.bias = torch.nn.parameter.Parameter(b4)

        for i in range(len(M1_w)):
            if i == 2 or i == 3:
                M1_w[i] = M1_w[i].transpose([1,0])
            elif i == 0 or i == 1:
                M1_w[i] = M1_w[i].transpose([2,3,1,0])


    def forward(self, x):
        c1_out = self.maxpool1(F.relu(self.conv1(x)))
        c2_out = self.maxpool2(F.relu(self.conv2(c1_out))).permute(0,2,3,1) 
        fc1_out = F.relu(self.fc1(c2_out.reshape(c2_out.size(0), -1)))
        fc2_out = F.relu(self.fc2(fc1_out))
        
        return [c1_out.permute(0,2,3,1), c2_out, fc1_out, fc2_out]

# obtain model 1 and model 2's new weights from the shared codebook
PQ_w1 = get_weight_from_codebook(c, i1, w1[3])
PQ_w2 = get_weight_from_codebook(c, i2, w2[3])

# compute and display the difference between well trained weights and new weights
m10_loss = get_weight_loss(PQ_w1[0], w1[0])
m11_loss = get_weight_loss(PQ_w1[1], w1[1])
m12_loss = get_weight_loss(PQ_w1[2], w1[2])

m20_loss = get_weight_loss(PQ_w2[0], w2[0])
m21_loss = get_weight_loss(PQ_w2[1], w2[1])
m22_loss = get_weight_loss(PQ_w2[2], w2[2])

m1_recon_loss = m10_loss, m11_loss, m12_loss
m2_recon_loss = m20_loss, m21_loss, m22_loss
print('\n----- Coodbook Reconstruction Loss -----')
print('M1 : ', m1_recon_loss)
print('M2 : ', m2_recon_loss)

# load train and test data
train_x1, train_y1, test_x1, test_y1 = load_sound20_data(config)
train_x2, train_y2, test_x2, test_y2 = load_fasion_data(config)
train_x1 = train_x1.transpose(0,3,1,2)
train_x2 = train_x2.transpose(0,3,1,2)
test_x1 = test_x1.transpose(0,3,1,2)
test_x2 = test_x2.transpose(0,3,1,2).astype('float64') 
total_test_size_1 = len(test_x1)
total_test_size_2 = len(test_x2)

# preprocess train and test data
class DATASET(Dataset):
    def __init__(self, x,y):        
        self.x = torch.from_numpy(x)
        self.y = torch.tensor(torch.from_numpy(y), dtype = torch.long)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
data1_train = DATASET(train_x1, train_y1)
data1_test = DATASET(test_x1, test_y1)
data2_train = DATASET(train_x2, train_y2)
data2_test = DATASET(test_x2, test_y2)
train_loader1 = torch.utils.data.DataLoader(dataset=data1_train, batch_size=config.batch_size, shuffle=True)
test_loader1 = torch.utils.data.DataLoader(dataset=data1_test, batch_size=config.batch_size, shuffle=False)
train_loader2 = torch.utils.data.DataLoader(dataset=data2_train, batch_size=config.batch_size, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(dataset=data2_test, batch_size=config.batch_size, shuffle=False)

# Check the acc of well trained networks and merged networks using test data
def Testing(test_loader1, test_loader2, total_size_1, total_size_2):
    
    correct_count_1 = 0
    correct_count_2 = 0
    for x,y in test_loader1:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        M1_out = M1.forward(x)  
        M1_out_pq = PQ1.forward(x)
        _ , index = torch.max(M1_out[3], 1)
        correct_count_1 += torch.sum(index==y).item()
        _ , index = torch.max(M1_out_pq[3], 1)
        correct_count_2 += torch.sum(index==y).item()
        
    accuracy1 = correct_count_1/total_size_1
    accuracy1_pq = correct_count_2/total_size_1
    
    correct_count_1 = 0
    correct_count_2 = 0
    for x,y in test_loader2:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        M2_out = M2.forward(x)  
        M2_out_pq = PQ2.forward(x)
        _ , index = torch.max(M2_out[3], 1)
        correct_count_1 += torch.sum(index==y).item()
        _ , index = torch.max(M2_out_pq[3], 1)
        correct_count_2 += torch.sum(index==y).item()
    
    accuracy2 = correct_count_1/total_size_2    
    accuracy2_pq = correct_count_2/total_size_2
    
    print('PQ_Sound     Model  Accuracy: %.5f'%accuracy1_pq)
    print('Well-trained Model1 Accuracy: %.5f'%accuracy1)
    print('PQ_Fasion    Model  Accuracy: %.5f'%accuracy2_pq)
    print('Well-trained Model2 Accuracy: %.5f'%accuracy2)  

# define networks, criterion and optimizer
if torch.cuda.is_available():
    M1 = Well_trained_M1(w1, b1).cuda()
    PQ1 = PQ_M1(PQ_w1, b1).cuda()
    criterion1 = nn.CrossEntropyLoss().cuda()
    optimizer1 = optim.Adam(PQ1.parameters(), lr=lr_rate)
    M2 = Well_trained_M2(w2, b2).cuda()
    PQ2 = PQ_M2(PQ_w2, b2).cuda()
    criterion2 = nn.CrossEntropyLoss().cuda()
    optimizer2 = optim.Adam(PQ2.parameters(), lr=lr_rate)
else:
    M1 = Well_trained_M1(w1, b1)
    PQ1 = PQ_M1(PQ_w1, b1)  
    criterion1 = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(PQ1.parameters(), lr=lr_rate)
    M2 = Well_trained_M2(w2, b2)    
    PQ2 = PQ_M2(PQ_w2, b2) 
    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = optim.Adam(PQ2.parameters(), lr=lr_rate)

# test performance of all networks before training
print('------ Test Before Calibration -----')
Testing(test_loader1, test_loader2, total_test_size_1, total_test_size_2)

# train merged network 1
print('\n----- Start Calibration -----\n')
total_iter = 0
done = False
while True:
    if done == True:
        break
    for i, (data, y1) in enumerate(train_loader1):
        if torch.cuda.is_available():
            data = data.cuda()
            y1 = y1.cuda()
        
        if total_iter == int(config.max_iter * 0.8):            
            for g in optimizer1.param_groups:
                g['lr'] = g['lr'] * 0.5
                
        optimizer1.zero_grad()
        M1_out = M1.forward(data)
        M1_out_pq = PQ1.forward(data) 
        M1_loss=0
        for i in range(4):
            M1_loss = M1_loss + (torch.mean(torch.abs(M1_out[i] - M1_out_pq[i])))
        M1_hardlabel = criterion1(M1_out_pq[3], y1) # put after data loader
        loss1 = M1_loss + 10*M1_hardlabel
        loss1.backward()
        optimizer1.step()
               
        total_iter += 1
        if total_iter == config.max_iter:
            done = True
            break 

# train merged network 2
total_iter = 0
done = False
while True:
    if done == True:
        break
    for i, (data, y2) in enumerate(train_loader2):
        if torch.cuda.is_available():
            data = data.cuda()
            y2 = y2.cuda()
        if total_iter == int(config.max_iter * 0.8):            
            for g in optimizer2.param_groups:
                g['lr'] = g['lr'] * 0.5
        
        optimizer2.zero_grad()
        M2_out = M2.forward(data)
        M2_out_pq = PQ2.forward(data) 
        M2_loss=0
        for i in range(4):
            M2_loss = M2_loss + (torch.mean(torch.abs(M2_out[i] - M2_out_pq[i])))
        M2_hardlabel = criterion2(M2_out_pq[3], y2) # put after data loader
        loss2 = M2_loss + 10*M2_hardlabel
        loss2.backward()
        optimizer2.step()
               
        total_iter += 1
        if total_iter == config.max_iter:
            done = True
            break   

# test performance of all networks after training
print('------ Test After Calibration -----')
Testing(test_loader1, test_loader2, total_test_size_1, total_test_size_2)

# compute and display the difference between well trained weights and new weights
m10_loss = get_weight_loss(PQ1.conv1.weight.cpu().detach().numpy().transpose(2,3,1,0), w1[0])
m11_loss = get_weight_loss(PQ1.conv2.weight.cpu().detach().numpy().transpose(2,3,1,0), w1[1])
m12_loss = get_weight_loss(PQ1.fc1.weight.cpu().detach().numpy().transpose(1,0), w1[2])

     
m20_loss = get_weight_loss(PQ2.conv1.weight.cpu().detach().numpy().transpose(2,3,1,0),\
                           M2.conv1.weight.cpu().detach().numpy().transpose(2,3,1,0))
m21_loss = get_weight_loss(PQ2.conv2.weight.cpu().detach().numpy().transpose(2,3,1,0),\
                           M2.conv2.weight.cpu().detach().numpy().transpose(2,3,1,0))
m22_loss = get_weight_loss(PQ2.fc1.weight.cpu().detach().numpy().transpose(1,0),\
                           M2.fc1.weight.cpu().detach().numpy().transpose(1,0))


m1_recon_loss = m10_loss.item(), m11_loss.item(), m12_loss.item()
m2_recon_loss = m20_loss.item(), m21_loss.item(), m22_loss.item()
print('\n----- Coodbook Reconstruction Loss -----')
print('M1 : ', m1_recon_loss)
print('M2 : ', m2_recon_loss)

print('\nTotal time taken: ', time.time()-start)
