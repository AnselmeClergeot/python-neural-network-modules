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

	def backward(self, X, Y, learningRate) :
		self.computeErrors(X, Y)
		self.updateWeights(X, learningRate)

	def setAllWeights(self, WList) :
		for i in range(len(self.layers)) :
			self.layers[i].W = WList

	def getAllWeights(self) :
		WList = []
		for i in range(len(self.layers)) :
			WList.append(self.layers[i].W)	

	def computeErrors(self, X, Y) :
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
		for i in range(len(self.layers)) :
			currentLayer = self.layers[i]

			inputsCol = np.append(np.asarray(X), 1).reshape(len(X)+1, 1)
			error = currentLayer.error.reshape(1, len(currentLayer.error))

			gradients = np.dot(inputsCol, error)

			currentLayer.W = np.add(currentLayer.W, np.multiply(learningRate, gradients))

			X = currentLayer.output
