import argparse

arg_lists = []
parser = argparse.ArgumentParser()
def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg
# change target dir
def setting():

    net_arg = add_argument_group('Calibration Setting')
    net_arg.add_argument('--data_ratio', type=float, default=0.1,
                        help='The percent of training data used in calibration stage')
    net_arg.add_argument('--use_1000_training', type=str2bool, default=False)
    net_arg.add_argument('--max_iter', type=int, default=100)
    net_arg.add_argument('--batch_size', type=int, default=64)
    net_arg.add_argument('--lr_rate', type=float, default=0.0001)
    net_arg.add_argument('--use_testing', type=str2bool, default=False)
    net_arg.add_argument('--random_seed', type=int, default=10)
    #net_arg.add_argument('--weight_dir', type=str, default='/home/iis/Desktop/NeuralMerger/weight&data/Well_trained_weight/')
    net_arg.add_argument('--weight_dir', type=str, default='C:\\Users\\LI HAOYANG\\Desktop\\research\\gan project\\NeuralMerger\\Experiment1\\Experiment1\\Merge_Lenet\\KMeans_stage\\') 
    config, unparsed = parser.parse_known_args()

    return config

