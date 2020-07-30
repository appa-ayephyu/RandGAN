import numpy as np
from operator import add, neg

def prune_the_support_set(generator_list, discriminator_list, generator_meta_strategy, discriminator_meta_strategy, meta_matrix):
    min_gen = min(generator_meta_strategy)
    min_dis = min(discriminator_meta_strategy)
    g_index = []
    d_index = []
    for i in range(len(generator_meta_strategy)):
        if generator_meta_strategy[i] == min_gen:
            g_index.append(i)

    for i in range(len(discriminator_meta_strategy)):
        if discriminator_meta_strategy[i] == min_dis:
            d_index.append(i)
    g_index = tuple(g_index)
    d_index = tuple(d_index)
    discriminator_list = np.delete(discriminator_list, d_index, 0)
    generator_list = np.delete(generator_list, g_index, 0)
    generator_meta_strategy = np.delete(generator_meta_strategy, g_index, 0)
    discriminator_meta_strategy = np.delete(discriminator_meta_strategy, d_index, 0)
    meta_matrix = np.delete(meta_matrix, g_index, 0)
    meta_matrix = np.delete(meta_matrix, d_index, 1)
    return generator_list, discriminator_list, generator_meta_strategy, discriminator_meta_strategy, meta_matrix


def termination_checking(generator_list, discriminator_list, generator_meta_strategy, discriminator_meta_strategy, meta_matrix):
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
    else:
        return False


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


def meta_solver(generator_list, discriminator_list, meta_matrix):
    rowcnt, colcnt, value_of_game = NE_solver(meta_matrix)
    local_generator_meta_strategy = np.array([i / 100 for i in rowcnt], dtype=float)
    local_discriminator_meta_strategy = np.array([c / 100 for c in colcnt], dtype=float)
    return local_generator_meta_strategy, local_discriminator_meta_strategy
