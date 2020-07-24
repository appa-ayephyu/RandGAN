import tensorflow as tf
import numpy as np
from data_loader import load_sound20_data, load_fasion_data
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

# function's purpose:
# to differentiate the cluster label number when there are more than one segemnt in a conv layer
# eg. if 2 segemnts are present in a layer with codeword = 64, cluster label number 0 to 63 will be given for segment 1, 
# 64 to 127 will be given for segment 2
        # range(3)
        # input1: (1,800) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 64 (num codeword per segment)
        # input2: (1,1600) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 128 (num codeword per segment)     
        # input3: (512,1024) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 64 (num codeword per segment)   
def embedding_idx(index,k_num):

    for i in range(index.shape[0]): # for every segment
        index[i] = index[i] + k_num*i # modify the label number when there are multiple segemnts. Eg. if 512 segments ar present and there
                                      # are 1024 different codewords, segemnt one's labels will be within 0 to 1023. Segment two's labels
                                      # will be within 1024 to 2047 and so on

    return index

#------------------------------------------------------------------------------------------------------------------------------------------

    # shared_codebook = [(1,64,1), (1,128,32), (512,64,8)] segment,num_codeword, shape of codeword
    # w1/w2 =  [(5, 5, 1, 32), (5, 5, 32, 64), (4096, 1024), (1024, 20/10) # original weights
    # b1/b2 =  [(32,), (64,), (1024,), (20/10,)]  original bias
    # i1/i2 =  [(1,800), (1,1600), (512,1024)] segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to)
    # align1/align2 = [(1,), (32,), (4096,)] 0 to chn in - 1
    # mean_adjust = [1,1,1] append 1 per merged layer
    # config = check setting
class LeNet:
    def __init__(self,shared_codebook, w1,b1,i1,align1,w2,b2,i2,align2,mean_adjust, config):

        self.config = config
        self.codebook  = shared_codebook
        self.mean_adjust = mean_adjust
        self.mean_adjust = list(self.mean_adjust)
        self.mean_adjust.append(1.) # increase from [1,1,1] t0 [1,1,1,1]
        print ("Mean Adjustment    : [%.1f, %.1f, %.1f, %.1f]"%(self.mean_adjust[0],self.mean_adjust[1],self.mean_adjust[2],self.mean_adjust[3])) #[1,1,1,1]
        print ("Calibration Iter   : %d"%config.max_iter) # 5000 training iter
        if config.use_1000_training: # how much training data to use
            print ("Calibration Data   : 1000")
        else:
            print ("Calibration Data_r : %.2f"%config.data_ratio)
        print ("Learning Rate      : %.4f"%config.lr_rate) # learning rate = 0.0001
        print ("Batch Size         : %d"%config.batch_size) # batch size = 64 

        self.c  = [] # define tf weight variable c: [ (64,1), (128,32), (32768,8) ] (total number of codework, shape of codeword) 
        for i in range(3): 
            self.c.append(tf.Variable(shared_codebook[i].reshape(shared_codebook[i].shape[0]*shared_codebook[i].shape[1],shared_codebook[i].shape[2]),dtype=tf.float32))
        
        self.x1     = tf.placeholder(tf.float32,[None,32,32,1])  # define tf network1's input. shape = [any_size, width,height,depth] 
        self.y1     = tf.placeholder(tf.int64) # define tf network1's true output label 
        self.w1     = w1 # [(5, 5, 1, 32), (5, 5, 32, 64), (4096, 1024), (1024, 20) # network 1's original weights
        self.b1     = b1 # [(32,), (64,), (1024,), (20,)]  network1's original bias
        self.i1     = i1 # [(1,800), (1,1600), (512,1024)] segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to)
        self.align1 = align1 # [(1,), (32,), (4096,)] 0 to chn in - 1
        
        # update each layer's cluster label if there is more than 1 segment in the layer.
        # eg. if 2 segemnts are present in a layer with codeword = 64, cluster label number 0 to 63 will be given for segment 1, 
        # 64 to 127 will be given for segment 2
        # range(3) : for the 3 merge layers
        # input1: (1,800) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 64 (num of codeword per segment)
        # input2: (1,1600) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 128 (num of codeword per segment)         # input3: (512,1024) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 64 (num of codeword per segment      
        for i in range(len(i1)):
            self.i1[i]     =   embedding_idx(i1[i],shared_codebook[i].shape[1]) 


        self.x2     = tf.placeholder(tf.float32,[None,32,32,1]) # define tf network2's input. shape = [any_size, width,height,depth] 
        self.y2     = tf.placeholder(tf.int64) # define tf network2's true output label 
        self.w2     = w2 # [(5, 5, 1, 32), (5, 5, 32, 64), (4096, 1024), (1024, 10) # network 2's original weights
        self.b2     = b2 # [(32,), (64,), (1024,), (10,)]  network2's original bias
        self.i2     = i2 # [(1,800), (1,1600), (512,1024)] segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to)
        self.align2 = align2 # [(1,), (32,), (4096,)] 0 to chn in - 1
        
        # update each layer's cluster label if there is more than 1 segment in the layer.
        # eg. if 2 segemnts are present in a layer with codeword = 64, cluster label number 0 to 63 will be given for segment 1, 
        # 64 to 127 will be given for segment 2
        # range(3) : for the 3 merge layers
        # input1: (1,800) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 64 (num of codeword per segment)
        # input2: (1,1600) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 128 (num of codeword per segment)         # input3: (512,1024) segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to), 64 (num of codeword per segment
        for i in range(len(i2)):
            self.i2[i]    =   embedding_idx(i2[i],shared_codebook[i].shape[1])
        
        # does nth
        for i in range(len(self.mean_adjust)):
            self.w2[i] = self.w2[i]*self.mean_adjust[i]
        
        # define learning rate and self.lr_updata which reduces learning rate by half
        # For each model, compute the loss between the output of each original model's layer and the merged layer
        # For each model, compute the loss between the final output of the merged model and the true labels 
        # define self.M1_train and self.M2_train, which is the optimizer(ADAM) and loss function (using the above 2 losses)
        # define self.accuracy1, self.accuracy2, which is the accuracy of the original model's prediction
        # define self.accuracy1_pq, self.accuracy2_pq, which is the accuracy of the merged model's prediction
        self.build_loss_function()
        
        # display test acc of merged models and outputs' difference between original and merged models at each layer before and after
        # training to show the effect of calibration training    
        # load the train and test data for each model. Train both the merged models in batches. shuffle the entire training set after every         # epoch   
        self.Trainer()
        
        
        self.save_for_speedtest()
        
        
 #--------------------------------------------------------------------------------------------------------------------------------------- 

       
        # standard CNN. Output at every conv layer is passed through a relu function followed by a max_pool_2x2 and returned
        #               Output at every fc layer is passed through a relu function and returned
        # Hence at the last output layer, raw value for each class is passed through relu function and returned instead of probability 
        # contructed using original well trained weights and ORIGINAL BIAS
        
        # before 1st conv2d: x1.shape = [n, 32, 32, 1]  w1[0].shape = [5, 5, 1, 32]   b1.shape = 32,
        # before 1st max_poop_2x2: input data shape = [n, 32, 32, 32] num of data,w,h,d
        # c1_out.shape = [n, 16, 16, 32]
        
        # before 2nd conv2d: c1_out.shape = [n, 16, 16, 32]  w1[1].shape = [5, 5, 32, 64]   b1.shape = 64,
        # before 2nd max_poop_2x2: input data shape = [n, 16, 16, 64] num of data,w,h,d
        # c2_out.shape = [n, 8, 8, 64]
        
        # before fc1_out: tf.contrib.layers.flatten(c2_out).shape = [n, 8*8*64=4096]  w1[2].shape = [4096, 1024]   b1[2] = [1024,]
        # fc1_out.shape = [n, 1024]
        
        # before fc2_out: fc1_out.shape = [n, 1024]    w1[3] = [1024, 20]    b1[3] = 20,
        # fc2_out.shape =  [n, 20]  (each 1,20 array has 20 raw values for the 20 class, not the probability of each class yet)
        
    def Well_trained_M1(self):
        c1_out          =  max_pool_2x2(tf.nn.relu(conv2d(self.x1, self.w1[0]) + self.b1[0]))
        c2_out          =  max_pool_2x2(tf.nn.relu(conv2d(c1_out, self.w1[1]) + self.b1[1]))
        fc1_out         =  tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(c2_out), self.w1[2]) + self.b1[2])
        fc2_out         =  tf.matmul(fc1_out, self.w1[3]) + self.b1[3]

        return [c1_out, c2_out, fc1_out, fc2_out] # c1_out.shape = [n, 16, 16, 32]
                                                  # c2_out.shape = [n, 8, 8, 64]
                                                  # fc1_out.shape = [n, 1024]
                                                  # fc2_out.shape =  [n, 20]    

#---------------------------------------------------------------------------------------------------------------------------------------


    # Re arrange the merged weight to make them readiliy fit into CNN models
    # Compute the mean difference between merged weights and original well trained weights. Store the loss as self.mij_loss
    # contruct standard CNN using merged weights and original bias
    # return the output from the CNN at each layer
    def PQ_M1(self):
        # c (m*k,r)  (6,128,8) from author
        # i (m,h*w*outc) (6,5*5*256) from author
        # out (m,h*w*outc,r) from author

#----------------------------------------------------------------------------------------------------------------------------------        
        # Re-structure merged weights + the orginal weights from the last unmerged fc layer to be the model's new weights 
        
        # c: [ (64,1), (128,32), (32768,8) ] (total number of codework, shape of codeword)
        # i1 = [(1,800), (1,1600), (512,1024)] segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to)
        
        # tf.nn.embedding_lookup(c,i1) = [ (1,800,1), (1,1600,32), (512,1024,8) ]  for every label in i1, replace label with codeword
        
        # transpose with 0,2,1 = [ (1,1,800), (1,32,1600), (512,8,1024) ] change arrange from segmnent,num codeword, codeword size
        #                                                                                  to segment, codeword size, num codeword
        
        # reshape for con layers: (i1[i].shape[0] * c[i].shape[1]), w, h, chn out
        # reshape for fc layers: (i1[i].shape[0] * c[i].shape[1]), chn out
        # to: [ 1(1*1), 5, 5, 32], [ 32(1*32), 5, 5, 64], [ 4096(512*8), 1024] 
        # new structure for conv: chn in,w,h,chn out   new structure for fc: chn in, chn out
        
        # transpose w0 = [5,5,1,32], w1 = [5,5,32,64], w2 = [4096,1024]   w,h,chn in, chn out/ chn in, chn out
        # w3 = [1024, 20] original weights from model1's last fc layer
        
        w0 = tf.transpose(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[0],self.i1[0]),[0,2,1]),[self.i1[0].shape[0]*int(self.c[0].shape[1]),5 ,5 ,32]),(1,2,0,3))        
        w1 = tf.transpose(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[1],self.i1[1]),[0,2,1]),[self.i1[1].shape[0]*int(self.c[1].shape[1]),5 ,5 ,64]),(1,2,0,3))
        w2 = tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[2],self.i1[2]),[0,2,1]),[self.i1[2].shape[0]*int(self.c[2].shape[1]),1024])
        w3 = tf.Variable(self.w1[3])
#------------------------------------------------------------------------------------------------------------------------------------
        
        # find the loss (a single value) between the original model's weights and the new merged weights 
        
        # self.w1: [(5, 5, 1, 32), (5, 5, 32, 64), (4096, 1024), (1024, 20)] # network 1's original weights
        # w0/1/2:  [(5, 5, 1, 32), (5, 5, 32, 64), (4096, 1024), (1024, 20)] # network 1's new weights
        # self.align  [(1,), (32,), (4096,)] 0 to chn in - 1
        
        # NOTE: tf.gather here does nth!!! 
        # tf.gather(self.w1[0],self.align1[0],axis=2) # extract entire tensor (include axis 0 and 1), for all the indices in
        # self.align1[0], along axis 2 of self.w1[0] Since w1[0]'s axis 2 has only indice 0, and self.align1[0] = [0], the result
        # tensor is still self.w1[0], hence
        # tf.gather(self.w1[0],self.align1[0],axis=2) = self.w1[0]
        # tf.gather(self.w1[1],self.align1[1],axis=2) = self.w1[1]
        # tf.gather(self.w1[2],self.align1[2],axis=0) = self.w1[2]
        
        # reduce_mean: return a single number. 
        # self.m1i_loss: the mean absolute difference between original weight and merged weight 
        
        self.m10_loss = tf.reduce_mean(tf.abs(tf.gather(self.w1[0],self.align1[0],axis=2)-w0))
        self.m11_loss = tf.reduce_mean(tf.abs(tf.gather(self.w1[1],self.align1[1],axis=2)-w1))
        self.m12_loss = tf.reduce_mean(tf.abs(tf.gather(self.w1[2],self.align1[2],axis=0)-w2))

#-------------------------------------------------------------------------------------------------------------------------------------
        # standard CNN. Output at every conv layer is passed through a relu function followed by a max_pool_2x2 and returned
        #               Output at every fc layer is passed through a relu function and returned
        # Hence at the last output layer, raw value for each class is passed through relu function and returned instead of probability 
        # contructed using MERGED weights and ORIGINAL BIAS
        
        
        # self.align1 [(1,), (32,), (4096,)] 0 to chn in - 1
        
        # before 1st conv2d: x1.shape = [n, 32, 32, 1]  w0.shape = [5, 5, 1, 32]   b1.shape = 32,
        # before 1st max_poop_2x2: input data shape = [n, 32, 32, 32] num of data,w,h,d
        # c1_out.shape = [n, 16, 16, 32]
        # tf.gather(c1_out,self.align1[1],axis=3): does nth
        
        # before 2nd conv2d: c1_out.shape = [n, 16, 16, 32]  w1.shape = [5, 5, 32, 64]   b1.shape = 64,
        # before 2nd max_poop_2x2: input data shape = [n, 16, 16, 64] num of data,w,h,d
        # c2_out.shape = [n, 8, 8, 64]
        
        # before fc1_out: tf.contrib.layers.flatten(c2_out).shape = [n, 8*8*64=4096]  w2.shape = [4096, 1024]   b1[2] = [1024,]
        # note: tf.gather does nothing again!
        # fc1_out.shape = [n, 1024]
        
        # before fc2_out: fc1_out.shape = [n, 1024]    w3 = [1024, 20]    b1[3] = 20,
        # fc2_out.shape =  [n, 20]  (each 1,20 array has 20 raw values for the 20 class, not the probability of each class yet)

        c1_out          =  max_pool_2x2(tf.nn.relu(conv2d(self.x1, w0) + self.b1[0]))
        c1_out          =  tf.gather(c1_out,self.align1[1],axis=3)

        c2_out          =  max_pool_2x2(tf.nn.relu(conv2d(c1_out, w1) + self.b1[1]))

        fc1_out         =  tf.nn.relu(tf.matmul(tf.gather(tf.contrib.layers.flatten(c2_out),self.align1[2],axis=1), w2) + self.b1[2])
        fc2_out         =  tf.nn.relu(tf.matmul(fc1_out, w3) + self.b1[3])

        return [c1_out, c2_out, fc1_out, fc2_out]
    
#--------------------------------------------------------------------------------------------------------------------------------------    

    # same as Well_trained_M1. The only difference is at the last fc
    # before fc2_out: fc1_out.shape = [n, 1024]    w2[3] = [1024, 10]    b2[3] = 10,
    # fc2_out.shape =  [n, 10]  (each 1,10 array has 10 raw values for the 10 class, not the probability of each class yet)
    def Well_trained_M2(self):
        c1_out          =  max_pool_2x2(tf.nn.relu(conv2d(self.x2, self.w2[0]) + self.b2[0]))
        c2_out          =  max_pool_2x2(tf.nn.relu(conv2d(c1_out, self.w2[1]) + self.b2[1]))
        fc1_out         =  tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(c2_out), self.w2[2]) + self.b2[2])
        fc2_out         =  tf.matmul(fc1_out, self.w2[3]) + self.b2[3]

        return [c1_out, c2_out, fc1_out, fc2_out] # c1_out.shape = [n, 16, 16, 32]
                                                  # c2_out.shape = [n, 8, 8, 64]
                                                  # fc1_out.shape = [n, 1024]
                                                  # fc2_out.shape =  [n, 10]        
    
    # same as PQ_M1:
    # Re arrange the merged weight to make them readiliy fit into CNN models
    # Compute the mean difference between merged weights and original well trained weights. Store the loss as self.mij_loss
    # contruct standard CNN using merged weights and original bias
    # return the output from the CNN at each layer
    # c1_out.shape = [n, 16, 16, 32]
    # c2_out.shape = [n, 8, 8, 64]
    # fc1_out.shape = [n, 1024]
    # fc2_out.shape =  [n, 10]
    def PQ_M2(self):
        # c (m*k,r)  (6,128,8)
        # i (m,h*w*outc) (6,5*5*256)
        # out (m,h*w*outc,r)
        w0 = tf.transpose(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[0],self.i2[0]),[0,2,1]),[self.i2[0].shape[0]*int(self.c[0].shape[1]),5 ,5 ,32]),(1,2,0,3))
        w1 = tf.transpose(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[1],self.i2[1]),[0,2,1]),[self.i2[1].shape[0]*int(self.c[1].shape[1]),5 ,5 ,64]),(1,2,0,3))
        w2 = tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[2],self.i2[2]),[0,2,1]),[self.i2[2].shape[0]*int(self.c[2].shape[1]),1024])
        w3 = tf.Variable(self.w2[3])

        self.m20_loss = tf.reduce_mean(tf.abs(tf.gather(self.w2[0],self.align2[0],axis=2)-w0))
        self.m21_loss = tf.reduce_mean(tf.abs(tf.gather(self.w2[1],self.align2[1],axis=2)-w1))
        self.m22_loss = tf.reduce_mean(tf.abs(tf.gather(self.w2[2],self.align2[2],axis=0)-w2))

        c1_out          =  max_pool_2x2(tf.nn.relu(conv2d(self.x2, w0) + self.b2[0]))
        c1_out          =  tf.gather(c1_out,self.align2[1],axis=3)

        c2_out          =  max_pool_2x2(tf.nn.relu(conv2d(c1_out, w1) + self.b2[1]))

        fc1_out         =  tf.nn.relu(tf.matmul(tf.gather(tf.contrib.layers.flatten(c2_out),self.align2[2],axis=1), w2) + self.b2[2])
        fc2_out         =  tf.nn.relu(tf.matmul(fc1_out, w3) + self.b2[3])

        return [c1_out, c2_out, fc1_out, fc2_out]
    
    #--------------------------------------------------------------------------------------------------------------------------------
    # define learning rate and self.lr_updata which reduces learning rate by half
    # For each model, compute the loss between the output of each original model's layer and the merged layer
    # For each model, compute the loss between the final output of the merged model and the true labels 
    # define self.M1_train and self.M2_train, which is the optimizer(ADAM) abd loss function (using the above 2 losses)
    # define self.accuracy1, self.accuracy2, which is the accuracy of the original model's prediction
    # define self.accuracy1_pq, self.accuracy2_pq, which is the accuracy of the merged model's prediction
    
    def build_loss_function(self):
        lr_rate         = tf.Variable(self.config.lr_rate,trainable = False) # define learning rate and make it untrainable
        self.lr_updata  = tf.assign(lr_rate,lr_rate*0.5) # reduce lr_rate by half when calling run(lr_updata)

        # M1 setting (Sound)
        # M1_out = [c1_out.shape = [n, 16, 16, 32], c2_out.shape = [n, 8, 8, 64], fc1_out.shape = [n, 1024], fc2_out.shape =  [n, 20] ]
        M1_out          = self.Well_trained_M1()  
        
        # M1_out_pq  = [c1_out, c2_out, fc1_out, fc2_out]
        # c1_out.shape = [n, 16, 16, 32]
        # c2_out.shape = [n, 8, 8, 64]
        # fc1_out.shape = [n, 1024]
        # fc2_out.shape =  [n, 20] 
        M1_out_pq       = self.PQ_M1()
        
        M1_loss=0
        
        # Find the mean difference in the output between original model1 and merged weight model1 at each layer,
        # then add up the differences. M1_loss is a single value.
        for i in range(4):
            M1_loss = M1_loss + (tf.reduce_mean(tf.abs(M1_out[i] - M1_out_pq[i])))
            
        # self.y1: network 1's true output labels
        # M1_out_pq[3]: Merged network 1's raw output values, passed through a relu
        # tf.nn.sparse_softmax_cross_entropy_with_logits: compute the cross entropy loss of the output from merged network vs true label,
        # for every data. sparse is used because the labels(self.y1) is not in one-hot format, but numeric label format. For example:
        # if self.y1 has 5 class, the each label a numeric value between 0 to 4, instead of a one hot vector like [0,0,0,1,0]. For one hot
        # label, sparse is not required.
        # M1_hard_label: the mean cross entropy loss between output of merged network and true label, across all data point.        
        M1_hardlabel    = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits (labels = self.y1,logits = M1_out_pq[3]))

        # define optimizer and loss function:
        # optimizer: AdamOptimizer
        # loss function: the mean difference in the output between original model1 and merged weight model1 at each layer,
 # then add up the differences + 10 * the mean cross entropy loss between output of merged network and true label, across all data point.
        if self.config.use_testing == False: #( False is default)
            self.M1_train        = tf.train.AdamOptimizer(lr_rate).minimize(M1_loss + 10*M1_hardlabel)
        else:
            self.M1_train        = tf.train.AdamOptimizer(lr_rate).minimize(M1_loss)

        # M1_out[3]/M1_out_pq[3]: [n,20] raw output from last fc layer passed through relu, for original model and merged weight  model
        # self.y1: network 1's true output labels
        # tf.nn.softmax(M1_out[3])/tf.nn.softmax(M1_out_pq[3]): convert the [n,20] raw output to [n,20] probabilty of each class
        # tf.argmax(softmax(),1): for every data point, obtain the class with the highest probabilty. output shape = (n,1), sec
        # dimension stores the predicted class label
        # tf.equal(argmax(), self.y1): compare how many rows of data match between predicted and actual label. Return a (n,1) array,
        # where second dimension's values are either 0 or 1.
        # tf.cast: changes dtype of the n*1 one hot array to float32
        # reduce_mean the float32 (n*1) array: the total number of 1 in the array's second dimension/n
        # accuracy1: an accuracy value between 0 and 1, for the well trained model's prediciton
        # accuracy1_pq: an accuracy value between 0 and 1, for the merged model's prediciton
        self.accuracy1          = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(M1_out[3]), 1), self.y1), tf.float32))
        self.accuracy1_pq       = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(M1_out_pq[3]), 1), self.y1), tf.float32))

        # same as M1, with the only difference the number of output classes
        # M2 setting (Fashion)
        M2_out          = self.Well_trained_M2()
        M2_out_pq       = self.PQ_M2()
        M2_loss=0
        for i in range(4):
            M2_loss = M2_loss + (tf.reduce_mean(tf.abs(M2_out[i] - M2_out_pq[i])))

        M2_hardlabel        = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits (labels = self.y2,logits = M2_out_pq[3]))

        if self.config.use_testing == False:
            self.M2_train        = tf.train.AdamOptimizer(lr_rate).minimize(M2_loss + 10*M2_hardlabel)
        else:
            print('Use Testing Data in Calibration')
            self.M2_train        = tf.train.AdamOptimizer(lr_rate).minimize(M2_loss)

        self.accuracy2    = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(M2_out[3]), 1), self.y2), tf.float32))
        self.accuracy2_pq = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(M2_out_pq[3]), 1), self.y2), tf.float32))
#----------------------------------------------------------------------------------------------------------------------------------        
    # display test acc of merged models and outputs' difference between original and merged models at each layer before and after
    # training to show the effect of calibration training
    
    # load the train and test data for each model. Train both the merged models in batches. shuffle the entire training set after every     
    # epoch    

    def Trainer(self):
        # tensorflow syntax for running a tf graph
        init            = tf.global_variables_initializer()
        self.sess       = tf.Session()
        self.sess.run(init)
        
        # self.m10_loss: the mean of the absolute difference between all of model1's conv layer1's original weight and merged weight
        # self.m11_loss: the mean of the absolute difference between all of model1's conv layer2's original weight and merged weight
        # self.m12_loss: the mean of the absolute difference between all of model1's fc layer1's original weight and merged weight
        # self.m20_loss: the mean of the absolute difference between all of model2's conv layer1's original weight and merged weight
        # self.m21_loss: the mean of the absolute difference between all of model2's conv layer2's original weight and merged weight
        # self.m22_loss: the mean of the absolute difference between all of model2's fc layer1's original weight and merged weight
        # m1_recon_loss = [self.m10_loss, self.m11_loss, self.m12_loss] # a list storing the loss at each of the 3 layer
        # m2_recon_loss = [self.m20_loss, self.m21_loss, self.m22_loss] # a list storing the loss at each of the 3 layer
        m1_recon_loss = self.sess.run([self.m10_loss,self.m11_loss,self.m12_loss])
        m2_recon_loss = self.sess.run([self.m20_loss,self.m21_loss,self.m22_loss])
        print('\n----- Coodbook Reconstruction Loss -----')
        print('M1 : ', m1_recon_loss)
        print('M2 : ', m2_recon_loss)

        #   Data_loader: load train and test data as numpy array
        
        #   self.train_x1 (1663, 32, 32, 1)
        #   self.train_y1 (1663,)
        #   self.test_x1  (3727, 32, 32, 1)
        #   self.test_y1  (3727,)

        #   self.train_x2 (5500, 32, 32, 1)
        #   self.train_y2 (5500,)
        #   self.test_x2  (10000, 32, 32, 1)
        #   self.test_y2  (10000,)        
        self.train_x1, self.train_y1, self.test_x1, self.test_y1 = load_sound20_data(self.config)
        self.train_x2, self.train_y2, self.test_x2, self.test_y2 = load_fasion_data(self.config)

        #   Test Before Calibration
        # print the accuracy of all 4 models using test data only
        # ac1/ac2: original models' acc
        # ac1_pq/ac2_pq: merged models' acc
        self.Testing()

        print('\n----- Start Calibration -----')
        #   Calibration
        data1_num = self.train_x1.shape[0]  # total number of training data for model 1
        data2_num = self.train_x2.shape[0]  # total number of training data for model 2
        batch_size= self.config.batch_size  # batch_size = 64
        max_iter = self.config.max_iter     # max_iter = 5000
        
        for iter in range(max_iter): # run 5000 iterations. This iteration is not epoch. This is because the number of input data and
                                     # hence the batch number for the 2 models are different. So the actual epoches ran by the 2 models
                                     # are different during the 5000 iterations.
                    
            e1 = iter%(data1_num/batch_size) # repeatly generate a sequence of number from 0 to batch_size - 1.
            e2 = iter%(data2_num/batch_size) # eg. if batch number is 5, e1 during the first 10 iterations will be 0,1,2,3,4,0,1,2,3,4
                                             # this e1 or e2 will be the batch number during each model's individual training epoch.
                
            if e1 == 0: # e1 is 0 when all batches are fed into the model once. Hence when e1 is 0, this is the start of a new epoch for
                        # model 1
                np.random.seed(iter) # set a different random seed for every epoch
                
                # shuffle the sequence of training data
                random_shuffle  = np.random.choice(data1_num,data1_num,replace = False) # from a range (0 to data1_num-1), randomly 
                                                                                        # generate data1_num numbers with no repeatition 
                self.train_x1        = np.take(self.train_x1,random_shuffle,axis=0) # along the first dimension of self.train_x1/y1,
                self.train_y1        = np.take(self.train_y1,random_shuffle,axis=0) # take the indices in random_shuffle to form a new                                                                                         # array
            
            
            if e2 == 0: # e2 is 0 when all batches are fed into the model once. Hence when e2 is 0, this is the start of a new epoch for
                        # model 2
                np.random.seed(iter) # set a different random seed for every epoch
                
                # shuffle the sequence of training data
                random_shuffle  = np.random.choice(data2_num,data2_num,replace = False) # from a range (0 to data2_num-1), randomly 
                                                                                        # generate data2_num numbers with no repeatition 
                self.train_x2        = np.take(self.train_x2,random_shuffle,axis=0) # along the first dimension of self.train_x2/y2,
                self.train_y2        = np.take(self.train_y2,random_shuffle,axis=0) # take the indices in random_shuffle to form a new                                                                                         # array
                
            if iter == (max_iter*4/5): # reduce learning rate by half after completing 80% of the total iterations 
                self.sess.run(self.lr_updata)
            
            # get 1 batch (64) data for each model's train_x and train_y
            t_x1 = self.train_x1[int(e1*batch_size):int((e1+1)*batch_size)]
            t_y1 = self.train_y1[int(e1*batch_size):int((e1+1)*batch_size)]
            t_x2 = self.train_x2[int(e2*batch_size):int((e2+1)*batch_size)]
            t_y2 = self.train_y2[int(e2*batch_size):int((e2+1)*batch_size)]
            
            # feed the 4 batches of data defined above into the 2 original and 2 merged models, obtain the loss function and
            # optimize (update weight) with the optimizer
            # self.M1_train        = tf.train.AdamOptimizer(lr_rate).minimize(M1_loss + 10*M1_hardlabel)
            # self.M2_train        = tf.train.AdamOptimizer(lr_rate).minimize(M2_loss + 10*M2_hardlabel)
            self.sess.run([self.M1_train,self.M2_train],feed_dict={self.x1:t_x1,self.y1:t_y1,self.x2:t_x2,self.y2:t_y2})
            
            # for every 10000 iteration, print the acurracy of each of the 2 merged models evaluated using training data
            if iter %10000==0:
                print('Calibration iter : %5d'%iter)
                self.Validation(t_x1,t_y1,t_x2,t_y2)
        
        #   Test After Calibration
        # print the accuracy of all 4 models using test data only
        # ac1/ac2: original models' acc
        # ac1_pq/ac2_pq: merged models' acc
        self.Testing()

        # self.m10_loss: the mean of the absolute difference between all of model1's conv layer1's original weight and merged weight
        # self.m11_loss: the mean of the absolute difference between all of model1's conv layer2's original weight and merged weight
        # self.m12_loss: the mean of the absolute difference between all of model1's fc layer1's original weight and merged weight
        # self.m20_loss: the mean of the absolute difference between all of model2's conv layer1's original weight and merged weight
        # self.m21_loss: the mean of the absolute difference between all of model2's conv layer2's original weight and merged weight
        # self.m22_loss: the mean of the absolute difference between all of model2's fc layer1's original weight and merged weight
        # m1_recon_loss = [self.m10_loss, self.m11_loss, self.m12_loss] # a list storing the loss at each of the 3 layer
        # m2_recon_loss = [self.m20_loss, self.m21_loss, self.m22_loss] # a list storing the loss at each of the 3 layer  
        m1_recon_loss = self.sess.run([self.m10_loss,self.m11_loss,self.m12_loss])
        m2_recon_loss = self.sess.run([self.m20_loss,self.m21_loss,self.m22_loss])
        print('\n----- Coodbook Reconstruction Loss -----')
        print('M1 : ', m1_recon_loss)
        print('M2 : ', m2_recon_loss)

#------------------------------------------------------------------------------------------------------------------------------------        
        
    # The purpose of this function is unclear. It seems to only do some formating to the 2 models' original shared codebook
    # and model1's original labels, while it does not do anything with model2's original labels, even though the labels 
    # are not the same for model1 and model2.
    # Also, it only stores the above data into .bin files. Hence I don't see any reason for the implementation of this fucntion

    def save_for_speedtest(self):
        #Save test speed bin
        final_codebook=[]
        final_index1=[]
        final_index2=[]
        
        # self.sess.run(self.c[i]): load the original codebook layer by layer
        # self.c = [ (64,1), (128,32), (32768,8) ] (for each layer, the total number of codework, shape of codeword) 
        # self.codebook = np.array([ [1,64,1],
        #                            [1,128,32],
        #                            [512,64,8]  ]) (for each layer, segment , number of codeword per segemnt, shape of codeword)
        # reshape from self.c to self.codebook
        # after reshaping: [ (1, 64, 1), (1, 128, 32), (512, 64, 8) ]
        for i in range(3):
            final_codebook.append(self.sess.run(self.c[i]).reshape(self.codebook[i].shape[0],self.codebook[i].shape[1],self.codebook[i].shape[2]))
        
        # self.i1     = i1  [(1,800), (1,1600), (512,1024)] segment, w * h * chn out (which cluster center each 1*1*r kernel belongs to)
        # final_index1 = [ (32,5,5,1), (64,5,5,1), (1024,512) ] reshaped to chn out, w, d, chn in / num of output neuron, num of segments
        final_index1.append(self.i1[0].reshape(self.i1[0].shape[0],5,5,32).transpose(3,1,2,0))
        final_index1.append(self.i1[1].reshape(self.i1[1].shape[0],5,5,64).transpose(3,1,2,0))
        final_index1.append(self.i1[2].transpose(1,0))
        
        
        lenet_c=[] # [64, 128, 64] codeword c (number of codewords per segment in a layer)
        lenet_r=[] # [1, 32, 8] r (depth of 1 segment)
        for i in range(3):
            lenet_c.append(self.codebook[i].shape[1])
            lenet_r.append(self.codebook[i].shape[2])
            
        # self.align2 = [(1,), (32,), (4096,)] 0 to chn in - 1
        # self.align2[2] = (4096,) 0 to 4095 
        # will return False
        shuffle = checkShuffle(self.align2[2])
        
        # create /bin_file if it is not present, otherwise does nth
        # write the following to text file: 1-PARAMETER.txt
        # codeword c shapes [64, 128, 64]
        # [1, 32, 8] r (depth of 1 segment)
        # shuffle = False
        # mean adjust = [1,1,1,1]
        newpath = r'./bin_file'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            text_file = open("bin_file/1-PARAMETER.txt", "w")
            text_file.write("Lenet K_dim: {}".format(lenet_r))
            text_file.write("\nLenet K_num: {}".format(lenet_c))
            text_file.write("\nSHUFFLE: {}".format(shuffle))            # Suffle must be False in speed testing!
            text_file.write("\nMean Adjust: {}".format(self.mean_adjust))
            text_file.close()
            
            print('Start write bin')
            # final_codebook: [ (1, 64, 1), (1, 128, 32), (512, 64, 8) ] (segment , number of codeword per segemnt, shape of codeword)
            # final_index1: [ (32,5,5,1), (64,5,5,1), (1024,512) ] chn out, w, d, chn in / num of output neuron, num of segments
            # 3 (number of merged layers)
            
# create 1 bin file for weight storage and 1 bin file for label storage, for each merge layer (ctrdLst for weight, asmtLst for label)
            output_kmeans(final_codebook,final_index1,3,'CONV',True,1)  #only need one model because these two networks are identical
            #output_kmeans(final_codebook,final_index2,3,'CONV',True,2) #only need one model because these two networks are identical

#-------------------------------------------------------------------------------------------------------------------------            
            
    # print the accuracy of all 4 models using test data only
    # ac1/ac2: original models' acc
    # ac1_pq/ac2_pq: merged models' acc
    def Testing(self):
        ac1     = np.mean(self.sess.run([self.accuracy1],feed_dict={self.x1:self.test_x1,self.y1:self.test_y1}))
        ac2     = np.mean(self.sess.run([self.accuracy2],feed_dict={self.x2:self.test_x2,self.y2:self.test_y2}))
        ac1_pq  = np.mean(self.sess.run([self.accuracy1_pq],feed_dict={self.x1:self.test_x1,self.y1:self.test_y1}))
        ac2_pq  = np.mean(self.sess.run([self.accuracy2_pq],feed_dict={self.x2:self.test_x2,self.y2:self.test_y2}))
        print('------ Test Before Calibration -----')
        print('PQ_Sound     Model  Accuracy: %.5f'%ac1_pq)
        print('Well-trained Model1 Accuracy: %.5f'%ac1)
        print('PQ_Fasion    Model  Accuracy: %.5f'%ac2_pq)
        print('Well-trained Model2 Accuracy: %.5f'%ac2)
    
    # print the acurracy of each of the 2 merged models evaluated using training data
    def Validation(self,x1,y1,x2,y2):
        ac1_pq  = np.mean(self.sess.run([self.accuracy1_pq],feed_dict={self.x1:x1,self.y1:y1}))
        ac2_pq  = np.mean(self.sess.run([self.accuracy2_pq],feed_dict={self.x2:x2,self.y2:y2}))
        print('--- Training Accuracy ---')
        print('PQ_Sound     Model  Accuracy: %.5f'%ac1_pq)
        print('PQ_Fasion    Model  Accuracy: %.5f'%ac2_pq)

        

# standrd convolution for image data (both grey scale and RGB) with no shrinking of output    
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') 

# reduce input data's width and height (second and third dimension)
# sample input: [n, 32, 32, 32] num_data, w, h, d
# setting both filer size and stride size to 2 along a dimension reduce that dimension by half 
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# create 1 bin file for weight storage and 1 bin file for label storage, for each merge layer (ctrdLst for weight, asmtLst for label)
# final_codebook: [ (1, 64, 1), (1, 128, 32), (512, 64, 8) ] (segment , number of codeword per segemnt, shape of codeword)
# final_index1: [ (32,5,5,1), (64,5,5,1), (1024,512) ] chn out, w, d, chn in / num of output neuron, num of segments
# layer_num_list = 3 (number of merged layers)
# type = 'CONV'
# update = True
# model_num = 1
def output_kmeans(weight_list,label_list,layer_num_list,type,update,model_num):
    
    # for each merge layer, execute if block
    for iter_layer in range(layer_num_list):
        #print(iter_layer)
        if(update):
            file_path_ctrdLst     = "./bin_file/bvlc_lenet_M_aCaF.ctrdLst_update.%02d" % (iter_layer+1 ) # for final codebook
            file_path_asmtLst     = "./bin_file/bvlc_lenet_M%d_aCaF.asmtLst_update.%02d" % (model_num,iter_layer+1 ) # for final_index1

        else:
            file_path_ctrdLst     = "./bin_file/bvlc_lenet_aCaF.ctrdLst.%02d" % (iter_layer+1 )
            file_path_asmtLst     = "./bin_file/bvlc_lenet_aCaF.asmtLst.%02d" % (iter_layer+1 )
        
 # write final codebook 1 element at a time :[ (1, 64, 1), (1, 128, 32), (512, 64, 8) ] to bin file, together with some info about the data
        data = np.array(weight_list[iter_layer],dtype=np.float32)
        print(data.shape)
        write_bin(file_path_ctrdLst,data)
        
 # write final_index1 1 element at a time: [ (32,5,5,1), (64,5,5,1), (1024,512) ] to bin file, together with some info about the data
        data = np.array((label_list[iter_layer]),dtype=np.uint8)+1
        print(data.shape)
        write_bin(file_path_asmtLst,data)

                


# WRITE WEIGHT/LABEL TO BIN FILE, TOGETHER WITH THE INFORMATION ON THE DATA'S SHAPE AND NUMBER OF DIMENSION 
# file_path: "./bin_file/bvlc_lenet_aCaF.ctrdLst.%02d" if weights or "./bin_file/bvlc_lenet_aCaF.asmtLst.%02d" if labels
# data_np: if weights: (1, 64, 1) (1, 128, 32) (512, 64, 8)
# data_np: if labels:  (32,5,5,1) (64,5,5,1) (1024,512) 
def write_bin(file_path,data_np):
    file_path   = file_path + '.bin'
    dim_np      = np.array((data_np.shape),dtype=np.int32) # dim_np: the shape of each element
    dim_num     = np.int32(len(dim_np)) # number of dimensions. eg: if shape is (1, 64, 1), dim_num = 3
    
    output_file = open(file_path, 'wb') # create and open the file
    output_file.write(dim_num) # write dim_num
    output_file.write(np.array(dim_np)) # write dim_np
    c_order = data_np.flags['C_CONTIGUOUS'] # c_order = true
    if(c_order): # write the weights/labels to the file
        output_file.write((data_np))
    else:
        data_np1 = data_np.copy(order='C')
        
        output_file.write((data_np1))
    output_file.close()
    return

# list1 = (4096,) 0 to 4095 
# return False
def checkShuffle(list1):
    for i in range(list1.shape[0]):
        if list1[i] != i:
            return True
    return False


            

            





        





    