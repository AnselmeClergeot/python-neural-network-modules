#coding: utf-8

from Layer import Layer
import numpy as np
import pickle

def normalize(x, maxX) :
	return x/float(maxX)

class Neural_Network :
	"""
		Class used to manipulate a feed-forward neural network.
	"""
		
	def __init__(self, *layerSizes) :
		"""
			The constructor takes N parameters, with N being the number of layers of the network (including input, hidden and output)
			Each parameter describes how many neurons they are in the current layer.
		"""
		self.layers = []
		self.gradients = []

		for i in range(1, len(layerSizes)) :
			self.layers.append(Layer(layerSizes[i-1], layerSizes[i]))

	def saveToFile(self, filePath) :
		"""
			Save the network weights to a file.
		"""
		weights = []
		for layer in self.layers :
			weights.append(layer.W)

		with open(filePath, "wb") as saveFile :
			pickle.dump(weights, saveFile)	

	def loadFromFile(self, filePath) :
		"""	
			Load the weights configuration we saved previously.
		"""
		with open(filePath, "rb") as saveFile :
			weights = pickle.load(saveFile)

			for i in range(len(self.layers)) :
				self.layers[i].W = weights[i]

	def forward(self, X) :
		"""
			Perform forward propagation of a specific input.
			Returns the last layer complete output.
		"""
		for layer in self.layers :
			X = layer.calculateOutput(X)

		return X

	def backward(self, X, Y, learningRate) :
		"""
			Perform backpropagation on the network.
			Backpropagation is the algorithm that corrects the network to expect better predictions.
			X is the input and Y is the expected output of the output layer for this input.
			Learning rate describes how fast the regression will be performed. 0 < learningRate < 1
		"""
		self.computeErrors(X, Y)
		self.updateWeights(X, learningRate)

	def getAlgorithmError(self, X, Y) :
		"""
			This function is used to check if the backprop' algorithm is correctely implemented.
			We compare the gradients calculated by our backprop' algorithm, to the 
				gradients calculated using (J(a+e) - J(a-e))/2e with J our error function and e a real number close to zero.

			To get an estimation of how similar our two gradients vector are, we compute : norm(difference of two gradients vector)/norm(sum of two gradients vector)
				The value obtained must be close to 1e-10, 1e-9 approximately.
				If the value obtained is way more bigger, then we wrongly implemented our backprop' algorithm.

			I used it and the implementation of backprop' on this module seems correct.
		"""
		gradientsApproximation = np.array([])
		gradients = np.array([])

		for layer in self.getGradientsApproximation(X, Y) :
			for value in layer :
				gradientsApproximation = np.append(gradientsApproximation, value)

		for layer in self.getGradients(X, Y) :
			for value in layer :
				gradients = np.append(gradients, value)

		return np.linalg.norm(gradientsApproximation - gradients) / np.linalg.norm(gradientsApproximation + gradients)

	def getGradientsApproximation(self, X, Y) :
		"""
			This function is a part of the getAlgorithmError() function.
			
			We approximate each gradient of each weight and then return the result.
		"""

		gradients = []

		epsilon = 1e-4

		for layer in self.layers :
			WShape = np.shape(layer.W)

			layerGradients = np.zeros(WShape)

			for y in range(WShape[0]) :
				for x in range(WShape[1]) :
					layer.W[y][x] -= epsilon
		
					self.forward(X)

					y1 = self.layers[-1].J(Y)

					layer.W[y][x] += 2*epsilon

					self.forward(X)

					y2 = self.layers[-1].J(Y)

					layer.W[y][x] -= epsilon
	
					layerGradients[y][x] = -(y2 - y1) / (2 * epsilon)

			gradients.append(layerGradients)

		return gradients

	def getGradients(self, X, Y) :
		"""
			This function returns the gradients of each weight calculated by the backpropagation algorithm.
		"""
		self.backward(X, Y, 0)
		return self.gradients

	def computeErrors(self, X, Y) :
		"""
			Part of backpropagation algorithm.
			We calculate the error of each neuron in each layer, starting from the output layer and going back to first hiden layer.
		"""
		outputLayer = self.layers[-1]
	
		y = self.forward(X)

		outputLayer.error = np.multiply( outputLayer.sigmoidDerivative(), Y - y)

		for i in reversed(range(0, len(self.layers)-1)) :
			currentLayer = self.layers[i]
			nextLayer = self.layers[i+1]

			weights = np.delete(nextLayer.W, -1, 0)
			errorCol = nextLayer.error.reshape(len(nextLayer.error), 1)
			
			layerDerivative = currentLayer.sigmoidDerivative().reshape(len(currentLayer.sigmoidDerivative()), 1)

			currentLayer.error = np.multiply(layerDerivative, np.dot(weights, errorCol))			

	def updateWeights(self, X, learningRate) :
		"""
			Part of backpropagation algorithm.
			Updates the weights of each neuron using the neurons errors we calculated.
		"""
		for i in range(len(self.layers)) :
			currentLayer = self.layers[i]

			inputsCol = np.append(np.asarray(X), 1).reshape(len(X)+1, 1)
			errorLine = currentLayer.error.reshape(1, len(currentLayer.error))

			gradients = np.dot(inputsCol, errorLine)

			self.gradients.append(gradients)

			currentLayer.W = np.add(currentLayer.W, np.multiply(learningRate, gradients))

			X = currentLayer.output
