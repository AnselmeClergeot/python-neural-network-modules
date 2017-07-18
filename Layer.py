#coding: utf-8

import numpy as np

class Layer :
	
	def __init__(self, inputsNumber, layerSize) :
		
		self.W = np.random.uniform(-1, 1, size = (inputsNumber + 1, layerSize))

	def calculateOutput(self, X) :

		dotProduct = np.dot(np.append(X, 1), self.W)
		
		self.output = self.sigmoid(dotProduct)

		return self.output

	def sigmoid(self, output) :
		return 1 / (1 + np.exp(-output))

	def sigmoidDerivative(self) :
		return self.output * (1 - self.output)
