#coding: utf-8

from Layer import Layer
import numpy as np

class Neural_Network :
	
	def __init__(self, *layerSizes) :
		self.layers = []

		for i in range(1, len(layerSizes)) :
			self.layers.append(Layer(layerSizes[i-1], layerSizes[i]))

	def forward(self, X) :
		for layer in self.layers :
			X = layer.calculateOutput(X)

		return X

	def computeGradients(self, X, Y) :
		y = self.forward(X)

		outputLayer = self.layers[-1]
		outputLayer.gradients = np.multiply(self.sigmoidDerivative(outputLayer.output), Y - y)

		for i in range(0, len(self.layers)-1) :
			currentLayer = self.layers[i]
			nextLayer = self.layers[i+1]

			nextLayerTransposedW = nextLayer.W.T[:]
			nextLayerTransposedW = np.delete(nextLayerTransposedW, -1, 1)

			currentLayer.gradients = np.multiply(self.sigmoidDerivative(currentLayer.output),np.dot(nextLayer.gradients, nextLayerTransposedW))

	def updateWeights(self, X, learningRate) :
		for i in range(len(self.layers)) :
			currentLayer = self.layers[i]

			transposedInput = np.asmatrix(np.append(X, 1)).T[:]

			currentLayer.W = np.add(currentLayer.W, np.multiply(learningRate, np.dot(transposedInput, np.asmatrix(currentLayer.gradients))))

			X = currentLayer.output

	def backward(self, X, Y, learningRate) :
		self.computeGradients(X, Y)
		self.updateWeights(X, learningRate)

	def sigmoidDerivative(self, output) :
		return 1 * (1 - output)
