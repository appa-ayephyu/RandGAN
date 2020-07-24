import argparse

arg_lists = []
parser = argparse.ArgumentParser()
def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def setting():

    net_arg = add_argument_group('Calibration Setting')
    net_arg.add_argument('--shuffle', type=str2bool, default=False)
    net_arg.add_argument('--mean_adjust', type=str2bool, default=False)

    config, unparsed = parser.parse_known_args()

    return config

