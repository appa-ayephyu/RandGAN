"""
@Project description
The main objective is to apply the double oracle method into GAN
Basically, we focus on computing the mixed Nash equilibrium of GAN
Compared with the previous pure Nash equilibrium, mixed Nash equilibrium always exists
With the randomization of mixed strategy, GAN may have a better ability against exploitation

@Author Xinrun Wang
@Date 10 Dec 2019
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
import random
from operator import add, neg

import logging
from logging.handlers import RotatingFileHandler

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)  # default is INFO

from pathlib import Path

Path('./logs').mkdir(parents=True, exist_ok=True)
fileHandler = RotatingFileHandler('./logs/do-gan.log', maxBytes=2 * 1024 * 1024, backupCount=1000)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--max_memory", type=int, default=1000, help="the max number of discriminators/generators stored")
parser.add_argument("--epsilon", type=float, default=0.00005, help="the criteria for convergence")
parser.add_argument("--alpha", type=float, default=0.999, help="the conformal parameter")
parser.add_argument("--h", type=float, default=0.001, help="time step factor")
parser.add_argument("--max_steps", type=int, default=300, help="the max number of the time step")
opt = parser.parse_args()
print(opt)

os.makedirs("images", exist_ok=True)
# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
adversarial_loss = torch.nn.BCELoss()  # Loss function


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self):
        self.model.load_state_dict(torch.load("./checkpoints/generator_0"))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self):
        self.model.load_state_dict(torch.load("./checkpoints/discriminator_0"))


def sample_from_distribution(distribution):
    return Categorical(distribution).sample(sample_shape=torch.Size)


def core_oracle():
    """
    Warm initialization of the generator/discriminator pool, compared with cold initialization
    Using the traditional method in GAN to find the first generator and discriminator
    :return:
    """
    global prev_gloss
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()


    if os.path.isfile('./checkpoints/discriminator_0') and os.path.isfile('./checkpoints/generator_0'):
        generator.load()
        discriminator.load()

        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()
    else:
        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))

        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        Tensor = torch.FloatTensor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----------
        #  Training
        # ----------
        for epoch in range(5):  # (opt.n_epochs):
            rootLogger.info(f"Epoch {epoch}/5")
            for i, (imgs, _) in enumerate(dataloader):
                # Adversarial ground truths
                # valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                # fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                valid = Tensor(imgs.size(0), 1).fill_(1.0).requires_grad_(False).to(device)
                fake = Tensor(imgs.size(0), 1).fill_(0.0).requires_grad_(False).to(device)

                # Configure input
                real_imgs = imgs.clone().detach().to(device)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
                # Sample noise as generator input
                z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(device)
                # Generate a batch of images
                gen_imgs = generator(z).to(device)
                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs).to(device), valid)
                g_loss.backward()
                optimizer_G.step()
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs).to(device), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()).to(device), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()


                rootLogger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )


                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/c%d.png" % batches_done, nrow=5, normalize=True)


    global generator_list, discriminator_list
    generator.save("./checkpoints/generator_0")
    discriminator.save("./checkpoints/discriminator_0")
    generator_list.append("./checkpoints/generator_0")
    discriminator_list.append("./checkpoints/discriminator_0")
    payoff = aug_payoff(generator, discriminator)
    return payoff


def train_oracles(metamatrix, iteration_num, generator_distribution, discriminator_distribution):
    """
    Find the best-response generator for the given mixed strategy of the meta-game
    :return:
    """
    global generator_list, discriminator_list
    limit = 5
    generator = Generator()
    if iteration_num >= limit:
        generator.model.load_state_dict(torch.load(generator_list[iteration_num % limit]))
    if cuda:
        generator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    Tensor = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_discrinator = Discriminator()
    if iteration_num >= limit:
        training_discrinator.model.load_state_dict(torch.load(discriminator_list[iteration_num % limit]))
    optimizer_TD = torch.optim.Adam(training_discrinator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    if cuda:
        training_discrinator.cuda()

    for j, (imgs, _) in enumerate(dataloader):  # 938 batch

        # Adversarial ground truths
        valid = Tensor(imgs.size(0), 1).fill_(1.0).requires_grad_(False).to(device)

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise as generator input

        # Generate a batch of images

        # sample the discriminator (train for the discriminator len)
        for d_index in range(len(discriminator_list) - 1):
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(device)
            gen_imgs = generator(z).to(device)
            optimizer_G.zero_grad()
            D = Discriminator()
            D.model.load_state_dict(torch.load(discriminator_list[d_index]))
            if cuda:
                D.cuda()
            d_res = D(gen_imgs).to(device)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(d_res, valid).to(device)  # * discriminator_distribution[d_index]
            g_loss.backward()
            optimizer_G.step()

        optimizer_G.zero_grad()
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(device)
        gen_imgs = generator(z).to(device)
        d_res = training_discrinator(gen_imgs).to(device)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(d_res, valid).to(device)  # * discriminator_distribution[d_index]
        g_loss.backward()
        optimizer_G.step()

        optimizer_TD.zero_grad()
        valid = Tensor(imgs.size(0), 1).fill_(1.0).requires_grad_(False).to(device)
        fake = Tensor(imgs.size(0), 1).fill_(0.0).requires_grad_(False).to(device)

        real_imgs = imgs.clone().detach().to(device)

        # sample a mixture of generated images (train for the generator len)
        for g_index in range(len(generator_list)):
            optimizer_TD.zero_grad()
            real_loss = adversarial_loss(training_discrinator(real_imgs).to(device), valid)
            G = Generator()
            G.model.load_state_dict(torch.load(generator_list[g_index]))
            if cuda:
                G.cuda()
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(device)
            gen_imgs = G(z).to(device)
            # Measure discriminator's ability to classify real from generated samples
            fake_loss = adversarial_loss(training_discrinator(gen_imgs.detach()).to(device),
                                         fake)  # * 1/len(generator_list)
            d_loss = (real_loss + fake_loss)
            d_loss.backward()
            optimizer_TD.step()

        real_loss = adversarial_loss(training_discrinator(real_imgs).to(device), valid)
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(device)
        gen_imgs = generator(z).to(device)
        # Measure discriminator's ability to classify real from generated samples
        fake_loss = adversarial_loss(training_discrinator(gen_imgs.detach()).to(device),
                                     fake)  # * 1/len(generator_list)
        d_loss = (real_loss + fake_loss)
        d_loss.backward()
        optimizer_TD.step()

        rootLogger.info(
            "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
            % (iteration_num, opt.n_epochs, j, 100, g_loss.item())
        )

        batches_done = iteration_num * len(dataloader) + j
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d%d.png" % (iteration_num, j), nrow=5, normalize=True)

    if iteration_num < limit:
        generator.save("./checkpoints/generator_%d" % (iteration_num + 1))
        generator_list.append("./checkpoints/generator_%d" % (iteration_num + 1))

        training_discrinator.save("./checkpoints/discriminator_%d" % (iteration_num + 1))
        discriminator_list.append("./checkpoints/discriminator_%d" % (iteration_num + 1))
    else:
        generator.save("./checkpoints/generator_%d" % (iteration_num % limit))
        training_discrinator.save("./checkpoints/discriminator_%d" % (iteration_num % limit))

    # augment the meta payoff matrix
    num_generator = len(generator_list)-1
    aug_col = np.zeros((num_generator, 1))
    for gen in range(num_generator):
        g = Generator()
        g.model.load_state_dict(torch.load(generator_list[gen]))
        if cuda:
            g.cuda()
        aug_col[gen] = aug_payoff(g, training_discrinator).detach().cpu().clone()
    metamatrix = np.column_stack((metamatrix, aug_col))

    num_discriminator = len(discriminator_list)
    aug_row = np.zeros((num_discriminator, 1))
    for dis in range(num_discriminator):
        discriminator = Discriminator()
        discriminator.model.load_state_dict(torch.load(discriminator_list[dis]))
        if cuda:
            discriminator.cuda()
        aug_row[dis] = aug_payoff(generator, discriminator).detach().cpu().clone()
    aug_row = aug_row.transpose()
    return np.row_stack((metamatrix, aug_row))


def discriminator_oracle(metamatrix, iteration_num, generator_distribution, discriminator_distribution):
    """
    Find the best-response discriminator for the given mixed strategy of the meta-game
    :return:
    """
    global generator_list, discriminator_list, fixed_z

    discriminator = Discriminator()
    if cuda:
        discriminator.cuda()
        adversarial_loss.cuda()

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_iterations = 750
    sample = random.sample(range(900), 750)
    gen_len = len(generator_list)

    # for i in range(training_iterations): #len(generator_distribution)*2):
    for j, (imgs, _) in enumerate(dataloader):
        if j not in sample:
            continue
        # Adversarial ground truths

        valid = Tensor(imgs.size(0), 1).fill_(1.0).requires_grad_(False).to(device)
        fake = Tensor(imgs.size(0), 1).fill_(0.0).requires_grad_(False).to(device)

        real_imgs = imgs.clone().detach().to(device)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(device)

        #sample the generator mixture
        for g_index in range(len(generator_list)):
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs).to(device), valid)
            G = Generator()
            G.model.load_state_dict(torch.load(generator_list[g_index]))
            if cuda:
                G.cuda()
            gen_imgs = G(z).to(device)
            # Measure discriminator's ability to classify real from generated samples
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()).to(device), fake)  # * 1/len(generator_list)
            d_loss = (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()

        rootLogger.info(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
            % (iteration_num, opt.n_epochs, j, training_iterations, d_loss.item())
        )

    discriminator.save("./checkpoints/discriminator_%d" % (iteration_num + 1))
    discriminator_list.append("./checkpoints/discriminator_%d" % (iteration_num + 1))

    return meta_matrix  # np.column_stack((metamatrix, aug_col))


def aug_payoff(generator, discriminator):
    """
    Augment the payoff matrix for the meta-game
    :param generator:
    :param discriminator:
    :return:
    """
    Tensor = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sum_loss = 0
    num_case = 0

    sample = random.sample(range(900), 3)
    for i, (imgs, _) in enumerate(dataloader):
        if i not in sample:
            continue
        valid = Tensor(imgs.size(0), 1).fill_(1.0).requires_grad_(False).to(device)
        fake = Tensor(imgs.size(0), 1).fill_(0.0).requires_grad_(False).to(device)
        real_imgs = imgs.clone().detach().to(device)
        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).to(device)
        # sample the discriminator
        gen_imgs = generator(z).to(device)
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs).to(device), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()).to(device), fake)
        d_loss = (real_loss + fake_loss)
        sum_loss = sum_loss + d_loss
        num_case = num_case + 1
        if num_case == 3:
            break
    return sum_loss / num_case


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
    """
    solve the meta-game for a Nash equilibrium
    :return:
    """
    global generator_list, discriminator_list
    # metagame = nash.Game(meta_matrix)
    #
    # equilibria = metagame.support_enumeration(non_degenerate=False)
    # went_in_loop = False
    # for eq in equilibria:
    #     generator_distribution = eq[0]
    #     discriminator_distribution = eq[1]
    #     break
    rowcnt, colcnt, value_of_game = NE_solver(meta_matrix)
    generator_distribution = np.array([i / 100 for i in rowcnt], dtype=float)
    discriminator_distribution = np.array([c / 100 for c in colcnt], dtype=float)
    return generator_distribution, discriminator_distribution


def termination_checking(meta_matrix):
    """
    Check whether the iteration can be terminated
    Basically, if the best-response we obtain cannot increase the utility with at least \epsilon
    we can terminate the algorithms
    :return:
    """
    global count, generator_distribution, discriminator_distribution
    (row, col) = meta_matrix.shape

    # I added this since there is no generator_distribution or discriminator distribution here yet
    if generator_distribution.size == 0 or discriminator_distribution.size == 0:
        num_support = len(generator_list)
        generator_distribution = np.random.rand(num_support, 1)
        generator_distribution = generator_distribution / sum(generator_distribution)
        discriminator_distribution = np.random.rand(num_support, 1)
        discriminator_distribution = discriminator_distribution / sum(discriminator_distribution)

    current_utility = 0
    for r in range(row - 1):
        for c in range(col - 1):
            current_utility = \
                current_utility + generator_distribution[r] * discriminator_distribution[c] * meta_matrix[r][c]
    row_increment = 0
    for c in range(col):
        row_increment = row_increment + discriminator_distribution[c] * meta_matrix[-1][c]
    col_increment = 0
    for r in range(row):
        col_increment = col_increment + generator_distribution[r] * meta_matrix[r][-1]
    row_increment = -row_increment - (-current_utility)
    col_increment = col_increment - current_utility
    rootLogger.info(f"Current Utility {current_utility}")
    rootLogger.info(f"Row Increment {row_increment}")
    rootLogger.info(f"Column Increment {col_increment}")
    if -1 * row_increment < 0 and col_increment < 0:
        # print("Current Utility", current_utility)
        count = count + 1
        return True
    else:
        count = 0
        return False


def housekeeping(generator_list, discriminator_list, meta_matrix, generator_distribution, discriminator_distribution):
    min_gen = min(generator_distribution)
    min_dis = min(discriminator_distribution)
    g_index = []
    d_index = []
    for i in range(len(generator_distribution)):
        if generator_distribution[i] == min_gen:
            g_index.append(i)
            if os.path.exists(generator_list[i]):
                os.remove(generator_list[i])

    for i in range(len(discriminator_distribution)):
        if discriminator_distribution[i] == min_dis:
            d_index.append(i)
            if os.path.exists(discriminator_list[i]):
                os.remove(discriminator_list[i])
    g_index = tuple(g_index)
    d_index = tuple(d_index)
    discriminator_list = np.delete(discriminator_list, d_index, 0)
    generator_list = np.delete(generator_list, g_index, 0)
    generator_distribution = np.delete(generator_distribution, g_index, 0)
    discriminator_distribution = np.delete(discriminator_distribution, d_index, 0)
    meta_matrix = np.delete(meta_matrix, g_index, 0)
    meta_matrix = np.delete(meta_matrix, d_index, 1)

    return generator_list.tolist(), discriminator_list.tolist(), meta_matrix, generator_distribution, discriminator_distribution


def construct_meta_matrix(meta_matrix):
    global generator_list, discriminator_list
    meta_matrix = np.zeros((len(generator_list), len(discriminator_list)))
    for gen in range(len(generator_list)):
        G = Generator()
        G.model.load_state_dict(torch.load(generator_list[gen]))
        if cuda:
            G.cuda()
        for dis in range(len(discriminator_list)):
            D = Discriminator()
            D.model.load_state_dict(torch.load(discriminator_list[dis]))
            if cuda:
                D.cuda()
            meta_matrix[gen][dis] = aug_payoff(G, D).detach().cpu().clone()
    return meta_matrix


def image_generation(generator_distribution, iteration_num):
    global generator_list, fixed_z
    Tensor = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = fixed_z.to(device)  # Tensor(np.random.normal(0, 1, (64, opt.latent_dim))).to(device)
    # if len(generator_distribution) > 1:
    #     g_index = -1 # np.where(generator_distribution == generator_distribution.max())[0][0]
    # else:
    #     g_index = 0
    limit = 5
    if iteration_num < limit:
        g_index = iteration_num + 1
    else:
        g_index = iteration_num % limit
    G = Generator()
    print(generator_list[g_index])
    G.model.load_state_dict(torch.load(generator_list[g_index]))
    if cuda:
        G.cuda()
    gen_imgs = G(z).to(device)
    save_image(gen_imgs.data[:25], "images/%d.png" % iteration_num, nrow=5, normalize=True)


generator_list = []
discriminator_list = []
fixed_z = torch.FloatTensor(np.random.normal(0, 1, (64, opt.latent_dim)))

count = 0
if __name__ == "__main__":

    meta_matrix = np.zeros((1, 1))  # row is generator, col is discriminator, the payoff is the loss of the dis
    # meta_matrix[0][0] = core_oracle()
    generator_distribution = np.array([1])
    discriminator_distribution = np.array([1])

    limit = 5
    for i in range(limit):
        generator_list.append("./checkpoints/generator_%d" % i)
        discriminator_list.append("./checkpoints/discriminator_%d" % i)
    for iteration_num in range(opt.max_steps):
        if iteration_num == 4:
            continue
        if iteration_num < 20:
            continue
        meta_matrix = train_oracles(meta_matrix, iteration_num, generator_distribution, discriminator_distribution)
        # print(meta_matrix)
        # meta_matrix = construct_meta_matrix(meta_matrix)
        if iteration_num % 100 == 0:
            rootLogger.info(meta_matrix)
        generator_distribution, discriminator_distribution = meta_solver(meta_matrix)
        # rootLogger.info(f"Generator Distribution {generator_distribution}")
        # rootLogger.info(f"Discriminator Distribution {discriminator_distribution}")
        # print(f"Generator Distribution {generator_distribution}")
        # print(f"Discriminator Distribution {discriminator_distribution}")
        image_generation(generator_distribution, iteration_num)
        termination_checking(meta_matrix)
        if count > 3:
            print("generator distribution", end=" ")
            print(generator_distribution)
            for glist in generator_list:
                print(glist)
            break
        if len(generator_list) > 10:
            generator_list, discriminator_list, meta_matrix, generator_distribution, discriminator_distribution = housekeeping(
                generator_list, discriminator_list, meta_matrix, generator_distribution, discriminator_distribution)
            generator_distribution = 1 / sum(generator_distribution) * generator_distribution
            discriminator_distribution = 1 / sum(discriminator_distribution) * discriminator_distribution