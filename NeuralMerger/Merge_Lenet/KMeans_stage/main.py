import tensorflow as tf 
import time 
import numpy as np
from sklearn.cluster import KMeans
from KMeans import *
import os
from config import *
config = setting()




m1_weight=[]
m2_weight=[]
m1 = np.load('Sound20_weight.npy', allow_pickle=True, encoding = 'bytes')
m2 = np.load('fashion_weight.npy', allow_pickle=True, encoding = 'bytes')

# from (5,5,1,32) to (1,5*5*32)
m1_weight.append(m1[0].transpose(2,0,1,3).reshape(m1[0].shape[2],m1[0].shape[3]*m1[0].shape[0]*m1[0].shape[1]))
m2_weight.append(m2[0].transpose(2,0,1,3).reshape(m2[0].shape[2],m2[0].shape[3]*m2[0].shape[0]*m2[0].shape[1]))
# from (5,5,32,64) to (32,5*5*64)
m1_weight.append(m1[2].transpose(2,0,1,3).reshape(m1[2].shape[2],m1[2].shape[3]*m1[2].shape[0]*m1[2].shape[1]))
m2_weight.append(m2[2].transpose(2,0,1,3).reshape(m2[2].shape[2],m2[2].shape[3]*m2[2].shape[0]*m2[2].shape[1]))
# (4096,1024) 
m1_weight.append(m1[4])
m2_weight.append(m2[4])




share_index1=[]
share_index2=[]

subspace        = [  1, 8, 8]
codewords       = [ 64,64,64]
Shuffle         = config.shuffle
Mean_adjust     = config.mean_adjust

layer_num = 3
mean_adjust=[]
M_codebook=[] 
M1_index=[]

newpath = r'./weight.(%d,%d,%d).(%d,%d,%d).%r.%r'%(subspace[0],subspace[1],subspace[2],codewords[0],codewords[1],codewords[2],Shuffle,Mean_adjust)
if not os.path.exists(newpath):
    os.makedirs(newpath)
    for iter in range(layer_num):
        if Mean_adjust:
            m1_abs = np.mean(np.abs(m1_weight[iter]))
            m2_abs = np.mean(np.abs(m2_weight[iter]))
            mean_adjust_value   = m1_abs/m2_abs
            if mean_adjust_value > 2:
                mean_adjust_value = mean_adjust_value
            else:
                mean_adjust_value = 1
        else:
            mean_adjust_value   = 1
        mean_adjust.append(mean_adjust_value)
        #inputs (sec cv layer):
        # m1_weight[i] = 32*1600
        # m2_weight[i] = 32*1600
        # subspace[i] = 8
        # codewords[i] = 64
        codebook,index1,index2,order1,order2 = random_train(m1_weight[iter],m2_weight[iter]*mean_adjust_value,subspace[iter],codewords[iter],Shuffle)

        np.save(newpath + '/Ex1_shared_codebook.%d.npy'%(iter+1),codebook)
        np.save(newpath + '/Ex1_M1_index.%d.npy'%(iter+1),index1)
        np.save(newpath + '/Ex1_M1_order.%d.npy'%(iter+1),order1.flatten())
        np.save(newpath + '/Ex1_M2_index.%d.npy'%(iter+1),index2)
        np.save(newpath + '/Ex1_M2_order.%d.npy'%(iter+1),order2.flatten())
        M_codebook.append(codebook)
        M1_index.append(index1)

        share_index1.append(index1)
        share_index2.append(index2)



    np.save(newpath + '/Ex1_mean_adjust.npy',mean_adjust)
    print('Conv1 sharing ratio')
    share_ratio(share_index1[0],share_index2[0],codewords[0])
    print('Conv2 sharing ratio')
    share_ratio(share_index1[1],share_index2[1],codewords[1])
    print('Fc1 sharing ratio')
    share_ratio(share_index1[2],share_index2[2],codewords[2])
else:
    print("The file name (same parameter) already exist!")
    print("If you want to run the same parameter. Delete or Change the old folder name.")









