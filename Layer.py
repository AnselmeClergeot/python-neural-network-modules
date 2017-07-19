#coding: utf-8

import numpy as np

class Layer :
	"""
		A layer contains a matrix that contains the weights concerning each neuron in the layer.
	"""
	def __init__(self, inputsNumber, layerSize) :
		"""
			THe constructor builds a inputsNumber+1 x layerSize matrix of random numbers from -1 to 1. These are the weights, randomly initialized.
		"""	
		self.W = np.random.uniform(-1, 1, size = (inputsNumber + 1, layerSize))

	def calculateOutput(self, X) :
		"""
			Calculate the output of the layer (array of each neuron output) from an input vector.
			Returns the output calculated.
		"""
		dotProduct = np.dot(np.append(X, 1), self.W)
		
		self.output = self.sigmoid(dotProduct)

		return self.output

	def sigmoid(self, output) :
		"""
			Returns the output vector with every output scaled through the sigmoid activation function.
		"""
		return 1 / (1 + np.exp(-output))

	def sigmoidDerivative(self) :
		"""
			Returns the derivative of the output vector.
		"""
		return self.output * (1 - self.output)

	def J(self, Y) :
		"""
			Output error function.
		"""
		return 0.5 * (np.sum(np.multiply(Y - self.output, Y - self.output)))
