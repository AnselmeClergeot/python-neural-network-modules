#coding: utf-8

from Neural_Network import *

NN = Neural_Network(2, 2, 1)
NN.loadFromFile("networkConfig")

t = [
	[0, 0],
	[1, 1],
	[0, 1],
	[1, 0]
]

e = [
	[0],
	[0],
	[1],
	[1]
]

#for i in range(50000) :
#	for j in range(4) :
#		NN.backward(t[j], e[j], 0.2)

for training in t :
	print(training)
	print(NN.forward(training))
	
NN.saveToFile("networkConfig")
