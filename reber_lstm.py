import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../reber-grammar/')))

import pickle
import time
import matplotlib.pyplot as plt
import reber
import symmetrical_reber
from lstm import *

letters = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

learning_rate = 0.1

"""train_path = '../reber-datasets/reber_train_2.4M.txt'
test_path = '../reber-datasets/reber_test_1M.txt'
automaton = reber.create_automaton()"""

train_path = '../reber-datasets/symmetrical_reber_train_2.4M.txt'
test_path = '../reber-datasets/symmetrical_reber_test_1M.txt'
automaton = symmetrical_reber.create_automaton(0.5)

def string_to_sequence(string):
    sequence = np.zeros((len(string), len(letters), 1))
    for i, letter in enumerate(string):
        sequence[i, letter_to_index[letter]] = 1
    return sequence

def train_reber(N):
	start_time = time.time()
	weights = Weights(len(letters), len(letters))
	f = open(train_path)
	t = []
	accuracies = []
	for i in range(N):
		if i % 1000 == 0:
			t.append(i)
			print(i)
			accuracies.append(accuracy(weights, 1000))
			print(accuracies[-1])
		string = f.readline().strip()
		sequence = string_to_sequence(string)
		network = LstmNetwork(weights, len(sequence)-1)
		network.learn(sequence[:-1], sequence[1:], learning_rate)
	f.close()
	plt.plot(t, accuracies)
	plt.xlabel('Nombre de chaines')
	plt.ylabel('Précision')
	plt.title("Courbe d'apprentissage avec un état caché à {} poids ({:.2f}s)".format(weights.dim_s, time.time() - start_time))
	plt.show()
	return weights

def predict_correctly(weights, string, threshold):
	sequence = string_to_sequence(string)
	network = LstmNetwork(weights, len(sequence)-1)
	result = network.propagate(sequence[:-1])
	cur_state = automaton.start
	for i, (x, y) in enumerate(zip(sequence[:-1], result)):
		cur_state = cur_state.next(string[i])
		predicted_transitions = {letters[j] for j, activated in enumerate(y > threshold) if activated}
		if set(predicted_transitions) != set(cur_state.transitions.keys()):
			return False
	return True

def accuracy(weights, N):
	f = open(test_path)
	c = 0
	for i in range(N):
		if i % 1000 == 0:
			print(i)
		string = f.readline().strip()
		if predict_correctly(weights, string, 0.3):
			c += 1
	return c / N

def test_reber(weights):
	string = 'BTSSXXVV'
	sequence = string_to_sequence(string)
	network = LstmNetwork(weights, len(sequence))
	result = network.propagate(sequence)
	for letter, y in zip(string, result):
		print(letter, [(l, p[0]) for l, p in zip(letters, y)])

if __name__ == '__main__':
	weights = train_reber(100000)
	# Save the weights
	pickle.dump(weights, open('reber.pickle', 'wb'))
	#weights = pickle.load(open('reber.pickle', 'rb'))
	#print(accuracy(weights, 100000))
	#test_reber(weights)