# Matthew Kim
# mtt.kim@mail.utoronto.ca
import xml.etree.ElementTree as ET
import os

from scipy.misc import imsave,imread,imresize,imshow
import numpy as np
from PIL import Image
import random

def main():
	cropandsquare()

def cropandsquare():
	#fruits = ['./apples/','./bananas/','./oranges/']

	training = []
	valid = []
	test = []
	training_labels = []
	valid_labels = []
	test_labels = []

	training, valid, test, training_labels, valid_labels, test_labels = getSets(0,'./oranges/', 400,50,50, training, valid, test, training_labels, valid_labels, test_labels)
	training, valid, test, training_labels, valid_labels, test_labels = getSets(1,'./apples/', 400,50,50, training, valid, test, training_labels, valid_labels, test_labels)
	training, valid, test, training_labels, valid_labels, test_labels = getSets(2,'./bananas/', 400,50,50, training, valid, test, training_labels, valid_labels, test_labels)
	'''training, valid, test, training_labels, valid_labels, test_labels = getSets(3,'./berry/', 100,30,30, training, valid, test, training_labels, valid_labels, test_labels)
	training, valid, test, training_labels, valid_labels, test_labels = getSets(3,'./beans/', 40,15,15, training, valid, test, training_labels, valid_labels, test_labels)
	training, valid, test, training_labels, valid_labels, test_labels = getSets(3,'./gourd/', 100,30,30, training, valid, test, training_labels, valid_labels, test_labels)
	training, valid, test, training_labels, valid_labels, test_labels = getSets(3,'./lettuce/', 100,29,29, training, valid, test, training_labels, valid_labels, test_labels)
	training, valid, test, training_labels, valid_labels, test_labels = getSets(3,'./cherries/', 60,20,20, training, valid, test, training_labels, valid_labels, test_labels)
	'''
	others = ['./']

	save = {
		'training_inputs': np.array(training),
		'valid_inputs': np.array(valid),
		'test_inputs': np.array(test),
		'training_labels': np.array(training_labels),
		'valid_labels': np.array(valid_labels),
		'test_labels': np.array(test_labels)
	}

	np.savez_compressed('regular_fruits.npz', **save)
	#np.savez_compressed('non_cat_fruits.npz',**save)

def cropandsquareTest():

	test = []
	test_labels = []

	test, test_labels = getTestSets(3,'./chairs/', 50, test,test_labels)
	test, test_labels = getTestSets(3,'./lemons/', 50, test,test_labels)
	test, test_labels = getTestSets(3,'./mangoes/', 50, test, test_labels)

	save = {
		'inputs': np.array(test),
		'labels': np.array(test_labels)
	}

	np.savez_compressed('clm_test.npz', **save)

def getSets(label,fruit,n_training,n_valid,n_test,o_training, o_valid, o_test, o_training_labels, o_valid_labels, o_test_labels):

	path = fruit + 'Annotation/'

	training = []
	valid = []
	test = []
	training_labels = []
	valid_labels = []
	test_labels = []

	counter = 0

	for file in os.listdir(path):
		# Pesky .DS_Store
		if(file != '.DS_Store'):
			tree = ET.parse(path + file)
			root = tree.getroot()
			elements = len(root)
			name = root[1].text

			image = fruit + name + '.JPEG'
			try:
				image = imread(image)
			except :
				continue

			# for bounding box in image
			for i in range (0,elements-5):
				# crop image
				xmin = int(root[5 + i][4][0].text)
				ymin = int(root[5 + i][4][1].text)
				xmax = int(root[5 + i][4][2].text)
				ymax = int(root[5 + i][4][3].text)
				cropped = image[ymin:ymax,xmin:xmax,:]
				width = cropped.shape[0]
				height = cropped.shape[1]

				# Make the image into a square padded with random pixel values
				new = np.array([])
				if (float(width) / float(height) < 2 and float(height) / float(width) < 2):
					if width > height:
						new = (np.random.rand(width,width,3) * 255).astype(int)
						#new = np.zeros((width,width,3))
						new[:,:height,:] = cropped
					elif height > width:
						new = (np.random.rand(height,height,3) * 255).astype(int)
						#new = np.zeros((height,height,3))
						new[:width,:,:] = cropped
					else:
						new = cropped
				else:
					continue

				# Resize image
				new = imresize(new, (32,32), interp='cubic')
				#new = new.reshape([10000,3])
				#new = new.reshape([100,100,3])
				#imsave('./processed/' + file + '_' + str(i) + '.JPEG',new)

				new = new.reshape(1024,3).astype(int)

				if(counter < n_training):
					training.append(new)
					training_labels.append(label)
				elif(counter < n_training + n_valid):
					valid.append(new)
					valid_labels.append(label)
				elif(counter < n_training + n_valid + n_test):
					test.append(new)
					test_labels.append(label)
				counter += 1

	print(fruit)
	print(counter)

	return o_training + training, o_valid + valid, o_test + test, o_training_labels + training_labels, o_valid_labels + valid_labels, o_test_labels + test_labels


def getTestSets(label,fruit,n_test, o_test, o_test_labels):

	path = fruit + 'Annotation/'

	test = []
	test_labels = []

	counter = 0

	for file in os.listdir(path):
		# Pesky .DS_Store
		if(file != '.DS_Store'):
			tree = ET.parse(path + file)
			root = tree.getroot()
			elements = len(root)
			name = root[1].text

			image = fruit + name + '.JPEG'
			try:
				image = imread(image)
			except :
				continue

			# for bounding box in image
			for i in range (0,elements-5):
				# crop image
				xmin = int(root[5 + i][4][0].text)
				ymin = int(root[5 + i][4][1].text)
				xmax = int(root[5 + i][4][2].text)
				ymax = int(root[5 + i][4][3].text)
				cropped = image[ymin:ymax,xmin:xmax,:]
				width = cropped.shape[0]
				height = cropped.shape[1]

				# Make the image into a square padded with random pixel values
				new = np.array([])
				if (float(width) / float(height) < 2 and float(height) / float(width) < 2):
					if width > height:
						#new = (np.random.rand(width,width,3) * 255).astype(int)
						new = np.zeros((width,width,3))
						new[:,:height,:] = cropped
					elif height > width:
						#new = (np.random.rand(height,height,3) * 255).astype(int)
						new = np.zeros((height,height,3))
						new[:width,:,:] = cropped
					else:
						new = cropped
				else:
					continue

				# Resize image
				new = imresize(new, (32,32), interp='cubic')
				#new = new.reshape([10000,3])
				#new = new.reshape([100,100,3])
				#imsave('./processed/' + file + '_' + str(i) + '.JPEG',new)

				new = new.reshape(1024,3).astype(int)

				if(counter < n_test):
					test.append(new)
					test_labels.append(label)
				else:
					break
				counter += 1

	print(fruit)
	print(counter)

	return o_test + test, o_test_labels + test_labels


			

if __name__ == "__main__":
	main()