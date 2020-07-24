#!/usr/bin/env python

import os, sys, time
import shutil
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.transforms as tfs
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from miscs.pgd import attack_Linf_PGD, attack_FGSM
from miscs.loss import *
from operator import add, neg
from torch.distributions import Categorical

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet_32')
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--start_width', type=int, default=4)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--root', type=str, default="/data1/cifar-10-batches-py")
parser.add_argument('--img_width', type=int, default=32)
parser.add_argument('--iter_d', type=int, default=5)
parser.add_argument('--out_f', type=str, default="ckpt.adv-5.32px-cifar10")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--starting_epoch', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--adv_steps', type=int, default=5)
parser.add_argument('--epsilon', type=float, default=0.0625)
parser.add_argument('--our_loss', action='store_true', default=False)
opt = parser.parse_args()

def load_models():
    if opt.model == "resnet_32":
        from gen_models.resnet_32 import ResNetGenerator
        from dis_models.resnet_32 import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass, bn=True)
    elif opt.model == "resnet_64":
        from gen_models.resnet_64 import ResNetGenerator
        from dis_models.resnet_64 import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_128":
        from gen_models.resnet_small import ResNetGenerator
        from dis_models.resnet_small import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_imagenet":
        from gen_models.resnet import ResNetGenerator
        from dis_models.resnet import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass, bn=True)
    else:
        raise ValueError(f"Unknown model name: {opt.model}")
    if opt.ngpu > 0:
        gen, dis = gen.cuda(), dis.cuda()
        gen, dis = torch.nn.DataParallel(gen, device_ids=range(opt.ngpu)), \
                torch.nn.DataParallel(dis, device_ids=range(opt.ngpu))
    else:
        raise ValueError("Must run on gpus, ngpu > 0")
    if opt.starting_epoch > 0:
        gen.load_state_dict(torch.load(f'./{opt.out_f}/gen_epoch_{opt.starting_epoch-1}.pth'))
        dis.load_state_dict(torch.load(f'./{opt.out_f}/dis_epoch_{opt.starting_epoch-1}.pth'))
    return gen, dis

def get_loss():
    return loss_nll, loss_nll

def make_optimizer(model, beta1=0, beta2=0.9):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(beta1, beta2))
    return optimizer

def make_dataset():
    # Small noise is added, following SN-GAN
    def noise(x):
        return x + torch.FloatTensor(x.size()).uniform_(0, 1.0 / 128)
    if opt.dataset == "cifar10":
        trans = tfs.Compose([
            tfs.RandomCrop(opt.img_width, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = CIFAR10(root=opt.root, train=True, download=True, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "dog_and_cat_64":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = ImageFolder(opt.root, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "dog_and_cat_128":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = ImageFolder(opt.root, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "imagenet":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = ImageFolder(opt.root, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")
    return loader

def train(gen, dis):
    global train_loader
    # buffer:
    # gaussian noise
    z = torch.FloatTensor(opt.batch_size, opt.nz).cuda()
    fixed_z = Variable(torch.FloatTensor(8 * 10, opt.nz).normal_(0, 1).cuda())
    # random label
    y_fake = torch.LongTensor(opt.batch_size).cuda()
    np_y = np.arange(10)
    np_y = np.repeat(np_y, 8)
    fixed_y_fake = Variable(torch.from_numpy(np_y).type(torch.LongTensor).cuda())
    # fixed label
    zeros = Variable(torch.FloatTensor(opt.batch_size).fill_(0).cuda())
    ones = Variable(torch.FloatTensor(opt.batch_size).fill_(1).cuda())
    # loss
    Ld, Lg = get_loss()
    # start training
    for epoch in range(70): #opt.starting_epoch, opt.starting_epoch + opt.max_epoch):
        for count, (x_real, y_real) in enumerate(train_loader):
            if count % opt.iter_d == 0:
                # update generator for every iter_d iterations
                gen.zero_grad()
                # sample noise
                z.normal_(0, 1)
                vz = Variable(z)
                y_fake.random_(0, to=opt.nclass)
                v_y_fake = Variable(y_fake)
                v_x_fake = gen(vz, y=v_y_fake)
                v_x_fake_adv = v_x_fake
                d_fake_bin, d_fake_multi = dis(v_x_fake_adv)
                # ones.data.resize_as_(d_fake_bin.data) changed this for error
                with torch.no_grad():
                    ones.resize_as_(d_fake_bin.data)
                loss_g = Lg(d_fake_bin, ones, d_fake_multi, v_y_fake, lam=0.5)
                loss_g.backward()
                opt_g.step()
                print(f'[{epoch}/{opt.max_epoch-1}][{count+1}/{len(train_loader)}][G_ITER] loss_g: {loss_g.item()}')
            # update discriminator
            dis.zero_grad()
            # feed real data
            x_real, y_real = x_real.cuda(), y_real.cuda()
            v_x_real, v_y_real = Variable(x_real), Variable(y_real)
            # find adversarial example, this is changed for error
            # ones.data.resize_(y_real.size())
            with torch.no_grad():
                ones.resize_(y_real.size())
            v_x_real_adv = attack_Linf_PGD(v_x_real, ones, v_y_real, dis, Ld, opt.adv_steps, opt.epsilon)
            d_real_bin, d_real_multi = dis(v_x_real_adv)
            # accuracy for real images
            positive = torch.sum(d_real_bin.data > 0).item()
            _, idx = torch.max(d_real_multi.data, dim=1)
            correct_real = torch.sum(idx.eq(y_real)).item()
            total_real = y_real.numel()
            # loss for real images
            loss_d_real = Ld(d_real_bin, ones, d_real_multi, v_y_real, lam=0.5)
            # feed fake data
            z.normal_(0, 1)
            y_fake.random_(0, to=opt.nclass)
            vz, v_y_fake = Variable(z), Variable(y_fake)
            with torch.no_grad():
                v_x_fake = gen(vz, y=v_y_fake)
            d_fake_bin, d_fake_multi = dis(v_x_fake.detach())
            # accuracy for fake images
            negative = torch.sum(d_fake_bin.data > 0).item()
            _, idx = torch.max(d_fake_multi.data, dim=1)
            correct_fake = torch.sum(idx.eq(y_fake)).item()
            total_fake = y_fake.numel()
            # loss for fake images
            if opt.our_loss:
                loss_d_fake = Ld(d_fake_bin, zeros, d_fake_multi, v_y_fake, lam=1)
            else:
                loss_d_fake = Ld(d_fake_bin, zeros, d_fake_multi, v_y_fake, lam=0.5)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()
            print(f'[{epoch}/{opt.max_epoch-1}][{count+1}/{len(train_loader)}][D_ITER] loss_d: {loss_d.item()} acc_r: {positive/total_real}, acc_r@1: {correct_real/total_real}, acc_f: {negative/total_fake}, acc_f@1: {correct_fake/total_fake}')
        # generate samples
        with torch.no_grad():
            fixed_x_fake = gen(fixed_z, y=fixed_y_fake)
            fixed_x_fake.data.mul_(0.5).add_(0.5)
        x_real.mul_(0.5).add_(0.5)
        save_image(fixed_x_fake.data, f'./{opt.out_f}/sample_epoch_{epoch}.png', nrow=8)
        save_image(x_real, f'./{opt.out_f}/real.png')
        # save model
        torch.save(dis.state_dict(), f'./{opt.out_f}/dis_epoch_{epoch}.pth')
        torch.save(gen.state_dict(), f'./{opt.out_f}/gen_epoch_{epoch}.pth')
        # change step size
        if (epoch + 1) % 50 == 0:
            opt.lr /= 2
            opt_g, opt_d = make_optimizer(gen), make_optimizer(dis)


def aug_payoff(gen, dis):
    global train_loader
    sum_loss = 0
    num_case = 0
    for count, (x_real, y_real) in enumerate(train_loader):
        # gaussian noise
        z = torch.FloatTensor(opt.batch_size, opt.nz).cuda()
        fixed_z = Variable(torch.FloatTensor(8 * 10, opt.nz).normal_(0, 1).cuda())
        # random label
        y_fake = torch.LongTensor(opt.batch_size).cuda()
        np_y = np.arange(10)
        np_y = np.repeat(np_y, 8)
        fixed_y_fake = Variable(torch.from_numpy(np_y).type(torch.LongTensor).cuda())
        # fixed label
        zeros = Variable(torch.FloatTensor(opt.batch_size).fill_(0).cuda())
        ones = Variable(torch.FloatTensor(opt.batch_size).fill_(1).cuda())
        # update discriminator
        dis.zero_grad()
        # feed real data
        x_real, y_real = x_real.cuda(), y_real.cuda()
        v_x_real, v_y_real = Variable(x_real), Variable(y_real)

        # loss
        Ld, Lg = get_loss()

        #calculating payoff
        x_real, y_real = x_real.cuda(), y_real.cuda()
        v_x_real, v_y_real = Variable(x_real), Variable(y_real)
        # find adversarial example, this is changed for error
        # ones.data.resize_(y_real.size())
        with torch.no_grad():
            ones.resize_(y_real.size())
        v_x_real_adv = attack_Linf_PGD(v_x_real, ones, v_y_real, dis, Ld, opt.adv_steps, opt.epsilon)
        d_real_bin, d_real_multi = dis(v_x_real_adv)
        # accuracy for real images
        positive = torch.sum(d_real_bin.data > 0).item()
        _, idx = torch.max(d_real_multi.data, dim=1)
        correct_real = torch.sum(idx.eq(y_real)).item()
        total_real = y_real.numel()
        # loss for real images
        loss_d_real = Ld(d_real_bin, ones, d_real_multi, v_y_real, lam=0.5)
        # feed fake data
        z.normal_(0, 1)
        y_fake.random_(0, to=opt.nclass)
        vz, v_y_fake = Variable(z), Variable(y_fake)
        with torch.no_grad():
            v_x_fake = gen(vz, y=v_y_fake)
        d_fake_bin, d_fake_multi = dis(v_x_fake.detach())
        # accuracy for fake images
        negative = torch.sum(d_fake_bin.data > 0).item()
        _, idx = torch.max(d_fake_multi.data, dim=1)
        correct_fake = torch.sum(idx.eq(y_fake)).item()
        total_fake = y_fake.numel()
        # loss for fake images
        if opt.our_loss:
            loss_d_fake = Ld(d_fake_bin, zeros, d_fake_multi, v_y_fake, lam=1)
        else:
            loss_d_fake = Ld(d_fake_bin, zeros, d_fake_multi, v_y_fake, lam=0.5)
        loss_d = loss_d_real + loss_d_fake
        sum_loss = sum_loss + loss_d
        num_case = num_case + 1
        if num_case == 3:
            break
    return sum_loss / num_case


def generator_oracle(gen, epoche):
    global generator_list, discriminator_list, discriminator_meta_strategy, train_loader, meta_matrix

    # buffer:
    # gaussian noise
    z = torch.FloatTensor(opt.batch_size, opt.nz).cuda()
    fixed_z = Variable(torch.FloatTensor(8 * 10, opt.nz).normal_(0, 1).cuda())
    # random label
    y_fake = torch.LongTensor(opt.batch_size).cuda()
    np_y = np.arange(10)
    np_y = np.repeat(np_y, 8)
    fixed_y_fake = Variable(torch.from_numpy(np_y).type(torch.LongTensor).cuda())
    # fixed label
    zeros = Variable(torch.FloatTensor(opt.batch_size).fill_(0).cuda())
    ones = Variable(torch.FloatTensor(opt.batch_size).fill_(1).cuda())
    # loss
    Ld, Lg = get_loss()
    # start training
    for epoch in range(2):
        for count, (x_real, y_real) in enumerate(train_loader):
            # update generator for every iter_d iterations
            gen.zero_grad()
            # sample noise
            z.normal_(0, 1)
            vz = Variable(z)
            y_fake.random_(0, to=opt.nclass)
            v_y_fake = Variable(y_fake)
            v_x_fake = gen(vz, y=v_y_fake)
            v_x_fake_adv = v_x_fake
            if len(discriminator_list) > 1:
                temp = torch.tensor(
                    (discriminator_meta_strategy.ravel() - discriminator_meta_strategy.min()) / (
                                discriminator_meta_strategy.max() - discriminator_meta_strategy.min()))
                d_index = Categorical(temp).sample()
            else:
                d_index = 0
            discriminator = discriminator_list[d_index]
            d_fake_bin, d_fake_multi = discriminator(v_x_fake_adv)
            # ones.data.resize_as_(d_fake_bin.data) changed this for error
            with torch.no_grad():
                ones.resize_as_(d_fake_bin.data)
            loss_g = Lg(d_fake_bin, ones, d_fake_multi, v_y_fake, lam=0.5)
            loss_g.backward()
            opt_g.step()
            print(f'[{epoch}/{opt.max_epoch - 1}][{count + 1}/{len(train_loader)}][G_ITER] loss_g: {loss_g.item()}')

        # generate samples
        with torch.no_grad():
            fixed_x_fake = gen(fixed_z, y=fixed_y_fake)
            fixed_x_fake.data.mul_(0.5).add_(0.5)
        x_real.mul_(0.5).add_(0.5)
        save_image(fixed_x_fake.data, f'./{opt.out_f}/sample_epoch_{epoche}.png', nrow=8)
        save_image(x_real, f'./{opt.out_f}/real.png')
        # save model
        torch.save(gen.state_dict(), f'./{opt.out_f}/gen_epoch_{epoche}.pth')

        #meta_matrix augment
        generator_list.append(gen)
        num_discriminator = len(discriminator_list)
        aug_row = np.zeros((num_discriminator, 1))
        for d_num in range(num_discriminator):
            aug_row[d_num] = aug_payoff(gen, discriminator_list[d_num]).detach().cpu().clone()
        meta_matrix = np.row_stack((meta_matrix, aug_row))


def discriminator_classifier_oracle(dis, epoche):
    global generator_list, discriminator_list, generator_meta_strategy, train_loader, meta_matrix

    # buffer:
    # gaussian noise
    z = torch.FloatTensor(opt.batch_size, opt.nz).cuda()
    fixed_z = Variable(torch.FloatTensor(8 * 10, opt.nz).normal_(0, 1).cuda())
    # random label
    y_fake = torch.LongTensor(opt.batch_size).cuda()
    np_y = np.arange(10)
    np_y = np.repeat(np_y, 8)
    fixed_y_fake = Variable(torch.from_numpy(np_y).type(torch.LongTensor).cuda())
    # fixed label
    zeros = Variable(torch.FloatTensor(opt.batch_size).fill_(0).cuda())
    ones = Variable(torch.FloatTensor(opt.batch_size).fill_(1).cuda())
    # loss
    Ld, Lg = get_loss()
    # start training
    for epoch in range(2):  # opt.starting_epoch, opt.starting_epoch + opt.max_epoch):
        for count, (x_real, y_real) in enumerate(train_loader):
           # update discriminator
            dis.zero_grad()
            # feed real data
            x_real, y_real = x_real.cuda(), y_real.cuda()
            v_x_real, v_y_real = Variable(x_real), Variable(y_real)
            # find adversarial example, this is changed for error
            # ones.data.resize_(y_real.size())
            with torch.no_grad():
                ones.resize_(y_real.size())
            v_x_real_adv = attack_Linf_PGD(v_x_real, ones, v_y_real, dis, Ld, opt.adv_steps, opt.epsilon)
            d_real_bin, d_real_multi = dis(v_x_real_adv)
            # accuracy for real images
            positive = torch.sum(d_real_bin.data > 0).item()
            _, idx = torch.max(d_real_multi.data, dim=1)
            correct_real = torch.sum(idx.eq(y_real)).item()
            total_real = y_real.numel()
            # loss for real images
            loss_d_real = Ld(d_real_bin, ones, d_real_multi, v_y_real, lam=0.5)
            # feed fake data
            z.normal_(0, 1)
            y_fake.random_(0, to=opt.nclass)
            vz, v_y_fake = Variable(z), Variable(y_fake)
            if len(generator_list) > 1:
                temp = torch.tensor(
                    (generator_meta_strategy.ravel() - generator_meta_strategy.min()) / (
                            generator_meta_strategy.max() - generator_meta_strategy.min()))
                g_index = Categorical(temp).sample()
            else:
                g_index = 0
            generator = generator_list[g_index]
            with torch.no_grad():
                v_x_fake = generator(vz, y=v_y_fake)
            d_fake_bin, d_fake_multi = dis(v_x_fake.detach())
            # accuracy for fake images
            negative = torch.sum(d_fake_bin.data > 0).item()
            _, idx = torch.max(d_fake_multi.data, dim=1)
            correct_fake = torch.sum(idx.eq(y_fake)).item()
            total_fake = y_fake.numel()
            # loss for fake images
            if opt.our_loss:
                loss_d_fake = Ld(d_fake_bin, zeros, d_fake_multi, v_y_fake, lam=1)
            else:
                loss_d_fake = Ld(d_fake_bin, zeros, d_fake_multi, v_y_fake, lam=0.5)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()
            print(
                f'[{epoche}/{opt.max_epoch - 1}][{count + 1}/{len(train_loader)}][D_ITER] loss_d: {loss_d.item()} acc_r: {positive / total_real}, acc_r@1: {correct_real / total_real}, acc_f: {negative / total_fake}, acc_f@1: {correct_fake / total_fake}')
        # save model
        torch.save(dis.state_dict(), f'./{opt.out_f}/dis_epoch_{epoche}.pth')
        # change step size
        if (epoche + 1) % 50 == 0:
            opt.lr /= 2
            opt_g, opt_d = make_optimizer(gen), make_optimizer(dis)

        #augment the meta_matrix
        discriminator_list.append(dis)
        num_generator = len(generator_list)
        aug_col = np.zeros((num_generator, 1))
        for g_num in range(num_generator):
            aug_col[g_num] = aug_payoff(generator_list[g_num], dis).detach().cpu().clone()
        meta_matrix = np.column_stack((meta_matrix, aug_col))


def NE_solver(payoff_matrix, iterations=100):
    'Return the oddments (mixed strategy ratios) for a given payoff matrix'
    transpose = list(zip(*payoff_matrix))
    numrows = len(payoff_matrix)
    numcols = len(transpose)
    row_cum_payoff = [0] * numrows
    col_cum_payoff = [0] * numcols
    colpos = list(range(numcols))
    rowpos = list(map(neg, range(numrows)))
    colcnt = [0] * numcols
    rowcnt = [0] * numrows
    active = 0
    for i in range(iterations):
        rowcnt[active] += 1
        col_cum_payoff = list(map(add, payoff_matrix[active], col_cum_payoff))
        active = min(list(zip(col_cum_payoff, colpos)))[1]
        colcnt[active] += 1
        row_cum_payoff = list(map(add, transpose[active], row_cum_payoff))
        active = -max(list(zip(row_cum_payoff, rowpos)))[1]
    value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations
    return rowcnt, colcnt, value_of_game


def meta_solver(meta_matrix):
    global generator_list, discriminator_list
    rowcnt, colcnt, value_of_game = NE_solver(meta_matrix)
    local_generator_meta_strategy = np.array([i / 100 for i in rowcnt], dtype=float)
    local_discriminator_meta_strategy = np.array([c / 100 for c in colcnt], dtype=float)
    return local_generator_meta_strategy, local_discriminator_meta_strategy


def termination_checking(meta_matrix):
    global generator_meta_strategy, discriminator_meta_strategy
    (row, col) = meta_matrix.shape

    # I added this since there is no generator_distribution or discriminator distribution here yet
    if generator_meta_strategy.size == 0 or discriminator_meta_strategy.size == 0:
        num_support = len(generator_list)
        generator_meta_strategy = np.random.rand(num_support, 1)
        generator_distribution = generator_meta_strategy / sum(generator_meta_strategy)
        discriminator_meta_strategy = np.random.rand(num_support, 1)
        discriminator_meta_strategy = discriminator_meta_strategy / sum(discriminator_meta_strategy)

    current_utility = 0
    for r in range(row - 1):
        for c in range(col - 1):
            current_utility = \
                current_utility + generator_meta_strategy[r] * discriminator_meta_strategy[c] * meta_matrix[r][c]
    row_increment = 0
    for c in range(col):
        row_increment = row_increment + discriminator_meta_strategy[c] * meta_matrix[-1][c]
    col_increment = 0
    for r in range(row):
        col_increment = col_increment + generator_meta_strategy[r] * meta_matrix[r][-1]
    row_increment = -row_increment - (-current_utility)
    col_increment = col_increment - current_utility
    if -1 * row_increment < 0 and col_increment < 0:
        return True


def prune_the_support_set():
    global generator_list, discriminator_list, generator_meta_strategy, discriminator_meta_strategy, meta_matrix
    min_gen = min(generator_meta_strategy)
    min_dis = min(discriminator_meta_strategy)
    g_index = []
    d_index = []
    for i in range(len(generator_meta_strategy)):
        if generator_meta_strategy[i] == min_gen:
            g_index.append(i)
            if os.path.exists(generator_list[i]):
                os.remove(generator_list[i])

    for i in range(len(discriminator_meta_strategy)):
        if discriminator_meta_strategy[i] == min_dis:
            d_index.append(i)
            if os.path.exists(discriminator_list[i]):
                os.remove(discriminator_list[i])
    g_index = tuple(g_index)
    d_index = tuple(d_index)
    discriminator_list = np.delete(discriminator_list, d_index, 0)
    generator_list = np.delete(generator_list, g_index, 0)
    generator_meta_strategy = np.delete(generator_meta_strategy, g_index, 0)
    discriminator_meta_strategy = np.delete(discriminator_meta_strategy, d_index, 0)
    meta_matrix = np.delete(meta_matrix, g_index, 0)
    meta_matrix = np.delete(meta_matrix, d_index, 1)


generator_list = []
discriminator_list = []
generator_meta_strategy = []
discriminator_meta_strategy = []
meta_matrix = np.zeros((1, 1))
# fixed dataset
train_loader = make_dataset()

if __name__ == "__main__":
    # first models
    gen, dis = load_models()
    # optimizers
    opt_g, opt_d = make_optimizer(gen), make_optimizer(dis)

    train(gen, dis, train_loader)

    generator_list.append(gen)
    discriminator_list.append(dis)
    meta_matrix[0][0] = aug_payoff(gen, dis)

    #load new models
    gen, dis = load_models()
    for epochs in range(70):
        generator_oracle(gen, epochs)
        discriminator_classifier_oracle(dis, epochs)
        generator_meta_strategy, discriminator_meta_strategy = meta_solver(meta_matrix)
        if termination_checking(meta_matrix):
            break
        if len(generator_list) > 5:
            prune_the_support_set()


