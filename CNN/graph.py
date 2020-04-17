import matplotlib.pyplot as plt
import sys
import numpy as np


def display(file):
	
	npz = np.load(file)

	displayPlot(npz['train_ce'],npz['valid_ce'], 'Cross Entropy')
	displayPlot(npz['train_acc'],npz['valid_acc'], 'Accuracy')


def displayPlot(train, valid, ylabel):

	plt.clf()
	train = np.array(train)
	valid = np.array(valid)
	plt.plot(train[:, 0], train[:, 1], 'b', label='Train')
	plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation')
	plt.xlabel('Epoch')
	plt.ylabel(ylabel)
	plt.legend()
	plt.show()

if __name__ == "__main__":
	file = sys.argv[1]

	display(file)
