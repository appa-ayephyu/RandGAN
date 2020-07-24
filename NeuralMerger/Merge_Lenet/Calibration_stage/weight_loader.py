import tensorflow as tf
import numpy as np
import sys



def load_weight_data(config):
    
    # load load well trained weights
    # structure: w h in out
    #(5, 5, 1, 32)
    #(32,)
    #(5, 5, 32, 64)
    #(64,)
    #(4096, 1024)
    #(1024,)
    #(1024, 20/10)
    #(20/10,)
    
    M1_target_weight_    = np.load(config.weight_dir + 'Sound20_weight.npy',  allow_pickle=True, encoding = 'bytes')
    M2_target_weight_    = np.load(config.weight_dir + 'fashion_weight.npy',  allow_pickle=True, encoding = 'bytes')
    
    # M_w is a list of 4 weights:
    #(5, 5, 1, 32)
    #(5, 5, 32, 64)    
    #(4096, 1024)    
    #(1024, 20/10)
    
    # M_b is a list of 4 ?:
    #(32,)    
    #(64,)    
    #(1024,)    
    #(20/10,)    
    M1_w = []
    M1_b = []
    for i in range(4):
        M1_w.append(M1_target_weight_[i*2])
        M1_b.append(M1_target_weight_[i*2+1])

    M2_w = []
    M2_b = []
    for i in range(4):
        M2_w.append(M2_target_weight_[i*2])
        M2_b.append(M2_target_weight_[i*2+1])
        
    # load merged weights, labels and order into lists. 3 items per list. Each item is the merged weight/labels/order of 1 layer:
    # m_codebook = [(1,64,1), (1,128,32), (512,64,8)] segment,num_codeword, shape of codeword
    # M_index = [(1,800), (1,1600), (512,1024)] segment, w * h * chn out
    # M_ch_order = [(1,), (32,), (4096,)] 0 to chn in - 1 
    # mean_adjust = [1,1,1] append 1 per merged layer
    M_codebook=[]
    M1_index  =[]
    M2_index  =[]
    M1_ch_order  =[]
    M2_ch_order  =[]
    for i in range(3):
        M_codebook.append(np.load('PQ_weight/Ex1_shared_codebook.%d.npy'%(i+1)))
        #print(M_codebook[i]).shape
        M1_index.append(np.load('PQ_weight/Ex1_M1_index.%d.npy'%(i+1)))
        #print(M1_index[i]).shape
        M2_index.append(np.load('PQ_weight/Ex1_M2_index.%d.npy'%(i+1)))
        #print(M2_index[i]).shape
        M1_ch_order.append(np.load('PQ_weight/Ex1_M1_order.%d.npy'%(i+1)))
        #print(M1_ch_order[i])
        M2_ch_order.append(np.load('PQ_weight/Ex1_M2_order.%d.npy'%(i+1)))
        #print(M2_ch_order[i])
    mean_adjust = np.load('PQ_weight/Ex1_mean_adjust.npy')
    print('----- Codebook  Parameter  Setting -----')
    print('Codebook  subspace : [%3d, %3d, %3d]'%(M_codebook[0].shape[2],M_codebook[1].shape[2],M_codebook[2].shape[2]))
    print('Codeworkd numbers  : [%3d, %3d, %3d]'%(M_codebook[0].shape[1],M_codebook[1].shape[1],M_codebook[2].shape[1]))
    
    # though array is returned, they still need to be access in the same way as list. 
    # eg. np.array(M_codebook).shape = 3     np.array(M_codebook)[0].shape = 1,64,1
    return np.array(M_codebook), np.array(M1_w), np.array(M1_b), np.array(M1_index), np.array(M1_ch_order), np.array(M2_w), np.array(M2_b), np.array(M2_index), np.array(M2_ch_order), np.array(mean_adjust)
