import numpy as np
from lstm import *

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',\
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',\
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ',\
            ',', '.', '?', ';', ':', "'", '"', \
             '-', '(', ')', '&', '!']

letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

def string_to_sequence(string):
    sequence = np.zeros((len(string), len(letters), 1))
    for i, letter in enumerate(string):
        sequence[i, letter_to_index[letter]] = 1
    return sequence

def letter_to_onehot(string):
    sequence = [0 for _ in letters]
    sequence[letter_to_index[string]] = 1
    return sequence

def sequence_to_string(sequence):
    return ''.join([letters[np.argmax(x)] for x in sequence])

def learn_shakespeare(path, N):
    dim_x = len(letters)
    dim_s = len(letters)
    weights = Weights(dim_s + 100, dim_x, -100)
    s_prev = np.zeros((dim_s,1))
    h_prev = np.zeros((dim_s,1))

    f = open(path)

    for i, line in zip(range(N), f):
        if i % 100 == 0:
            print(i)
        sequence = string_to_sequence(line.strip().upper())
        if len(sequence) > 1:
            X = sequence[:-1]
            Y = sequence[1:]
            network = LstmNetwork(weights, len(sequence)-1)
            network.learn(X, Y)

    f.close()
    return weights

def sample(weights, n):
    s = 'T'
    x = string_to_sequence(s)[0]
    network = LstmNetwork(weights, n)
    result = network.propagate_self_feeding(x)
    print(sequence_to_string(result))


if __name__ == '__main__':
    #weights = learn_shakespeare('data_shakespeare_karpathy.txt', 40000)
    #pickle.dump(weights, open('shakespeare_s.pickle', 'wb'))
    weights = pickle.load(open('shakespeare_s.pickle', 'rb'))
    sample(weights, 100)