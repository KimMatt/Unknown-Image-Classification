import numpy as np
import sys
from cnn import CNNForward 
from nn import Softmax

def main(model, data):
	model = np.load(model)
	foreign = np.load(data)
	inputs = foreign['inputs']
	target = foreign['labels']
	Evaluate(inputs, target, model, CNNForward)

def Evaluate(inputs, target, model, forward, batch_size=-1,part35=False):
	"""Evaluates the model on inputs and target.

	Args:
	    inputs: Inputs to the network.
	    target: Target of the inputs.
	    model:  Dictionary of network weights.
	"""
	thresholds = [0.5,0.6,0.7,0.8,0.9]
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
		prediction = Softmax(forward(model, x)['y'])
		#ce += -np.sum(t * np.log(prediction))
		#acc += (np.argmax(prediction, axis=1) == np.argmax(t, axis=1)).astype('float').sum()

		''' Check thresholds
        '''
		if(step == num_steps -1):
			#targets = np.argmax(t,axis=1)
			results = []

			guesses = np.argmax(prediction,axis=1)
			print(prediction)
			values = [each[guesses[index]] for index, each in enumerate(prediction)]

			for threshold in thresholds:
				classified = 0
				for index, each in enumerate(values):
					if each > threshold:
						# example, probability, guessed class, target
						#results.append((x[index],each,guesses[index],targets[index]))
						classified += 1
				print("threshold: " + str(threshold))
				print("classified: " + str(classified) + ' total: ' + str(inputs.shape[0]))
				print("ratio: " + str(float(classified) / float(inputs.shape[0])))           	
				print("----------------------------------------------")

			#results = {
			#    'results' : results
			#}
			#print("Saving results")
			#Save('eval.npz', results)

	ce /= num_cases
	acc /= num_cases
	return ce, acc



if __name__ == "__main__":

	main(sys.argv[1], sys.argv[2])