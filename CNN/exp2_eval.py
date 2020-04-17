from __future__ import division
import sys
import numpy as np
from nn import Softmax
from cnn import CNNForward

def main(model, data):
	model = np.load(model)
	foreign = np.load(data)
	inputs = foreign['inputs']
	target = foreign['labels'].tolist()
	target_1hot = np.zeros([4, len(target)])

	for ii, xx, in enumerate(target):
		target_1hot[xx,ii] = 1.0

	ce, acc = Evaluate(inputs, target_1hot.T, model, CNNForward)
	print("ce: " + str(ce))
	print("acc: " + str(acc))

def Evaluate(inputs, target, model, forward, batch_size=-1,part35=False):
	"""Evaluates the model on inputs and target.

	Args:
	    inputs: Inputs to the network.
	    target: Target of the inputs.
	    model:  Dictionary of network weights.
	"""
	num_cases = inputs.shape[0]
	if batch_size == -1:
		batch_size = num_cases
	num_steps = int(np.ceil(num_cases / batch_size))
	ce = 0.0
	acc = 0.0
	for step in range(num_steps):
		start = step * batch_size
		end = min(num_cases, (step + 1) * batch_size)
		x = inputs[start: end]
		t = target[start: end]
		prediction = np.array(Softmax(forward(model, x)['y']))
		ce += -np.sum(t * np.log(prediction))
		print(np.argmax(t, axis=1))
		print(np.argmax(prediction, axis=1))
		acc += (np.argmax(prediction, axis=1) == np.argmax(
		    t, axis=1)).astype('float').sum()

	ce /= num_cases
	acc /= num_cases
	return ce, acc


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
