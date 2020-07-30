#!/usr/bin/env python

import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.models import inception_v3
from miscs.inception_score import inception_score as score
from torch.distributions import Categorical

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet_32')
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--nimgs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--start_width', type=int, default=4)
parser.add_argument('--splits', type=int, default=10)
parser.add_argument('--ngpu', type=int, default=1)
opt = parser.parse_args()

assert opt.nimgs % opt.splits == 0, "ERR: opt.nimgs must be divided by opt.splits"
assert (opt.nimgs // opt.splits) % opt.batch_size == 0, "ERR: opt.nimgs//opt.splits \
        must be divided by opt.batch_size"


def load_model():
    if opt.model == "resnet_32":
        from gen_models.resnet_32 import ResNetGenerator
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
    elif opt.model == "resnet_64":
        from gen_models.resnet_64 import ResNetGenerator
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
    elif opt.model == "resnet_128":
        from gen_models.resnet_small import ResNetGenerator
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
    else:
        raise ValueError(f"Unknown model name: {opt.model}")
    if opt.ngpu > 0:
        gen = gen.cuda()
        gen = torch.nn.DataParallel(gen, device_ids=range(opt.ngpu))
    else:
        raise ValueError("Must run on gpus, ngpu > 0")
    return gen


def load_inception():
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.cuda()
    inception_model = torch.nn.DataParallel(inception_model, \
            device_ids=range(opt.ngpu))
    inception_model.eval()
    return inception_model


def load_models():
    gen_array = []
    path_array = ["gen1.pth", "gen2.pth", "gen3.pth", "gen4.pth", "gen5.pth"]
    for i in range(5):
        gen = load_model()
        gen.load_state_dict(torch.load(path_array[i]))
        gen_array.append(gen)
    return gen_array


def gen_imgs(generator_meta_strategy):
    gens = load_models()
    # buffer:
    # gaussian noise
    z = torch.FloatTensor(opt.batch_size, opt.nz).cuda()
    # random label
    y_fake = torch.LongTensor(opt.batch_size).cuda()
    imgs = []
    with torch.no_grad():
        for i in range(0, opt.nimgs, opt.batch_size):
            z.normal_(0, 1)
            y_fake.random_(0, to=opt.nclass)
            v_z = Variable(z)
            v_y_fake = Variable(y_fake)
            temp = torch.tensor(
                (generator_meta_strategy.ravel() - generator_meta_strategy.min()) / (
                            generator_meta_strategy.max() - generator_meta_strategy.min()))
            d_index = Categorical(temp).sample()
            gen = gens[d_index]
            x_fake = gen(v_z, y=v_y_fake)
            x = x_fake.data.cpu().numpy()
            imgs.append(x)
    imgs = np.asarray(imgs, dtype=np.float32)
    nb, b, c, h, w = imgs.shape
    imgs = imgs.reshape((nb * b, c, h, w))
    return imgs, (h, w) != (299, 299)


def calc_inception(generator_meta_strategy):
    imgs, resize = gen_imgs(generator_meta_strategy)
    model = load_inception()
    mean_score, std_score = score(model, imgs, opt.batch_size, \
            resize, opt.splits)
    return mean_score, std_score


def main():
    generator_meta_strategy = [0.2, 0.2, 0.2, 0.2, 0.2]
    mean, std = calc_inception(generator_meta_strategy)
    print(f"Mean: {mean}, Std: {std}")


if __name__ == "__main__":
    main()
