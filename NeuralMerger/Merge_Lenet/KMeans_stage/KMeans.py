import tensorflow as tf 
import time 
import numpy as np
import os
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.nan)
import sys
np.set_printoptions(threshold=sys.maxsize)
import scipy.stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def hungarian(weight1,weight2,k_num):
    #weight1.shape[0] >= weight2.shape[0]
    weight1_ = weight1[:weight2.shape[0],:]
    hunga_input=[]
    for i in range(weight1_.shape[0]):
        for j in range(weight2.shape[0]):
            hunga_input.append(data_distance(weight1[i],weight2[j],k_num))

    
    hunga_input = np.array(hunga_input).reshape(weight1_.shape[0],weight2.shape[0])
    row_ind, col_ind = linear_sum_assignment(hunga_input)

    return np.arange(weight1.shape[0]),col_ind

def data_distance(data1,data2,k_num):

    data1_number = data1.shape[0]
    data2_number = data2.shape[0]
    max_num = np.max(data1)
    min_num = np.min(data1)
    if np.max(data2) > max_num:
        max_num = np.max(data2)
    if np.min(data2) < min_num:
        min_num = np.min(data2)
    
    
    bin_1,_ = np.histogram(data1,bins=k_num,range=(min_num,max_num),density=False)
    bin_2,_ = np.histogram(data2,bins=k_num,range=(min_num,max_num),density=False)
    bin_1   = bin_1/float(data1_number)
    bin_2   = bin_2/float(data2_number)
    score   = 0.
    for i in range(k_num):
        score = score + np.minimum(bin_1[i],bin_2[i])

    return (1-score)


def share_ratio(index_array1,index_array2,k_num):

    ratio =[]
    for i in range(index_array1.shape[0]):
        num = len(set(index_array1[i])) + len(set(index_array2[i]))
        ratio.append((num-k_num)/float(k_num))

    out = np.mean(np.array(ratio))
    print(out)
    
# inputs eg: ( 0:r-1, 1600 ), ( 0:r-1, 1600 ), k_num = 64, 1600
def kmeans(weight1,weight2,k_num,m1_num):
    
    weight1 = weight1.transpose(1,0) # 1600,r
    weight2 = weight2.transpose(1,0) # 1600,r
    #print(np.mean(np.abs(weight1)))
    #print(np.mean(np.abs(weight2)))
    
    data_       = np.concatenate([weight1,weight2],axis=0) # 3200,r

    kmeans      = KMeans(n_clusters=k_num ,tol=1e-6).fit(data_)

    clusters    = kmeans.cluster_centers_ # 64 cluster centers of dimension 1*r
    index       = kmeans.labels_ # 3200 labels that each 1*r vector belongs to
    '''
    pq_matrix   = np.take(clusters, index,axis=0)

    loss1       = np.mean(np.abs(weight1-pq_matrix[:m1_num,:]))
    loss2       = np.mean(np.abs(weight2-pq_matrix[m1_num:,:]))
    '''
    pq_matrix1   = np.take(clusters, index[:m1_num],axis=0) # ?
    pq_matrix2   = np.take(clusters, index[m1_num:],axis=0) # ?

    loss1       = np.mean(np.abs(weight1-pq_matrix1)) # ?
    loss2       = np.mean(np.abs(weight2-pq_matrix2)) # ?
    print([loss1,loss2])

    return clusters, index[:m1_num], index[m1_num:] # 64 cluster centers of dimension 1*r, # 1600 labels that each 1*r vector belongs to,
                                                    # for the 2 conv layers respectively

        # eg inputs:
        # m1_weight[i] = 32*1600
        # m2_weight[i] = 32*1600
        # subspace[i] = 8
        # codewords[i] = 64
def random_train(m1_layer_w,m2_layer_w,k_dim,k_num,shuffle):
    #print(m1_layer_w.shape)
    #print(m2_layer_w[0])
    out_index1      = []
    out_index2      = []
    out_codebook    = []
    out_order1      = []
    out_order2      = []
    m1_ch           = m1_layer_w.shape[0] # 32 (in chn/depth of kernel)
    m1_output_num   = m1_layer_w.shape[1] # 1600 (total number of 1*1*d weights of the layer)
    order = np.arange(m1_ch) # [0:31]
    for m in range(int(m1_ch/k_dim)): # d/r (number of 1*1*r vectors from 1*1*d vectors)
        print('%d / %d'%(m,m1_ch/k_dim)) 

        if shuffle == True:
            order1, order2 = hungarian(m1_layer_w[m*k_dim:(m+1)*k_dim], m2_layer_w[m*k_dim:(m+1)*k_dim], k_num)
            order1= order1 + m*k_dim
            order2 = order2 + m*k_dim
        else:
            order1 = order[m*k_dim:(m+1)*k_dim] # [0:r-1]
            order2 = order[m*k_dim:(m+1)*k_dim] # [0:r-1]
        
        # input: ( 0:r-1, 1600 ), ( 0:r-1, 1600 ), k_num = 64, 1600
        # output: 64 cluster centers of dimension 1*r, # 1600 labels that each 1*r vector belongs to,
                # for the 2 conv layers respectively
        codebook, index1, index2 = kmeans(np.take(m1_layer_w,order1,axis=0), np.take(m2_layer_w,order2,axis=0), k_num, m1_output_num)

        out_codebook.append(np.copy(codebook))
        out_index1.append(np.copy(index1))
        out_index2.append(np.copy(index2))
        out_order1.append(np.copy(order1))
        out_order2.append(np.copy(order2))
    

# out_codebook[i]: codewords for each segment of this conv layer eg. 64,r 
# out_index1[i]: labels of all 1*1*r vector for each segment in this conv layer for model1 eg. 1600,
# out_index1[i]: labels of all 1*1*r vector for each segment in this conv layer for model2 eg. 1600,
# out_order[i]: eg. r = 3, d = 9. out_order: [0,1,2], [3,4,5] , [7,8,9] (along the depth direction, the index of each segment of this
# conv layer for model1/2
    return np.array(out_codebook),np.array(out_index1),np.array(out_index2),np.array(out_order1),np.array(out_order2)


    