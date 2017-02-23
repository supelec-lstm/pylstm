import numpy as np
from lstm import *

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',\
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',\
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ',\
            ',', '.', '?', ';', ':', "'", '"', \
             '-', '(', ')', '&', '!']

letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

learning_rate = 0.1

def string_to_sequence(string):
    sequence = np.zeros((len(string), len(letters), 1))
    for i, letter in enumerate(string):
        if not letter in letters:
            return []
        sequence[i, letter_to_index[letter]] = 1
    return sequence

def letter_to_onehot(string):
    sequence = [0 for _ in letters]
    sequence[letter_to_index[string]] = 1
    return sequence

def sequence_to_string(sequence):
    return ''.join([letters[np.argmax(x)] for x in sequence])

def sample_sequence_to_string(sequence):
    return ''.join([letters[np.random.choice(len(letters), p=x.flatten())] for x in sequence])

def learn_shakespeare(path, N):
    dim_x = len(letters)
    dim_s = len(letters) + 50
    dim_y = len(letters)
    weights = Weights(dim_s, dim_x, dim_y)
    #s_prev = np.zeros((dim_s, 1))
    #h_prev = np.zeros((dim_s, 1))

    for _ in range(1000):
        f = open(path)
        for i, line in zip(range(N), f):
            if i % 100 == 0:
                print(i)
            if i % 1000 == 0:
                sample(weights, 20)
                print(weights.wg)
            sequence = string_to_sequence(line.strip().upper())
            if len(sequence) > 1:
                X = sequence[:-1]
                Y = sequence[1:]
                network = LstmNetwork(weights, len(sequence)-1)
                network.learn(X, Y, learning_rate)
        f.close()
    return weights

def sample(weights, n):
    for i in range(ord('A'), ord('Z')+1):
        s = chr(i).upper()
        x = string_to_sequence(s)[0]
        network = LstmNetwork(weights, n)
        result = network.propagate_self_feeding(x)
        print(chr(i) + sample_sequence_to_string(result))
        print(chr(i) + sequence_to_string(result))


if __name__ == '__main__':
    weights = learn_shakespeare('shakespeare/shakespeare_karpathy.txt', 40000)
    pickle.dump(weights, open('shakespeare.pickle', 'wb'))
    #weights = pickle.load(open('shakespeare_s.pickle', 'rb'))
    sample(weights, 10)