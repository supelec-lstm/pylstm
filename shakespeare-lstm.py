import numpy as np
from lstm import *

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',\
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',\
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ',\
            ',', '.', '?', ';', ':', "'", '"', '[', ']',\
             '-', '(', ')', '&', '!']

letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

def string_to_sequence(string):
    sequence = np.zeros((len(string), len(letters), 1))
    for i, letter in enumerate(string):
        sequence[i,letter_to_index[letter]] = 1
    return sequence

def letter_to_onehot(string):
    sequence = [0 for _ in letters]
    sequence[letter_to_index[string]] = 1
    return sequence

def onehot_to_letter(onehot):
    for i in range(len(letters)):
        if onehot[0][i]:
            return index_to_letter[i]

def indice_max(l):
    m = 0
    i_m = 0
    for i, val in enumerate(l):
        if m < val:
            m = val
            i_m = i
    return i_m

def learn_shakespeare(path, N):
    dim_x = len(letters)
    dim_s = len(letters)
    weights = Weights(dim_s, dim_x)
    s_prev = np.zeros((dim_s,1))
    h_prev = np.zeros((dim_s,1))

    f = open(path)
    string = f.readline().strip().upper()
    network = LstmNetwork_shakespeare_style(dim_s, dim_x, len(string))
    network.learn(string_to_sequence(string), string_to_sequence(string[0]), s_prev, h_prev)
    s_prev = network.cells[-1].s
    h_prev = network.cells[-1].h

    for i in range(N):
        if N%500 == 0:
            network = LstmNetwork_shakespeare_style(dim_s, dim_x, 100)
            print([onehot_to_letter(network.propagate(string_to_sequence('E'), np.zeros((dim_s,1)),np.zeros((dim_s,1)))[i]) for i in range()])
        string = f.readline().strip().upper()
        network = LstmNetwork_shakespeare_style(dim_s, dim_x, len(string))
        network.learn(string_to_sequence(string), h_prev, s_prev, h_prev)

        s_prev = network.cells[-1].s
        h_prev = network.cells[-1].h

    f.close()
