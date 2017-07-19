#coding: utf-8

from Neural_Network import *

NN = Neural_Network(2, 3, 1)

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

for i in range(5000) :
	for j in range(4) :
		NN.backward(t[j], e[j], 0.4)

for training in t :
	print(training)
	print(NN.forward(training))
	
