import numpy as np
from model import *
from weight_loader import load_weight_data
from config import *
import os
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = setting()

newpath = r'./bin_file'
#if not os.path.exists(newpath):
M_codebook, M1_w, M1_b, M1_index, M1_ch_order, M2_w, M2_b, M2_index, M2_ch_order,mean_adjust = load_weight_data(config)
trainer = LeNet(M_codebook, M1_w, M1_b, M1_index, M1_ch_order, M2_w, M2_b, M2_index, M2_ch_order,mean_adjust,config)
#else:
#    print("The binfile folder name already exist!")
#    print("Delete or Change the old folder name.")
