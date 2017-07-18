#coding: utf-8

from Neural_Network import *

NN = Neural_Network(2, 3, 1)

examples = [
	[0, 0],
	[1, 1],
	[0, 1],
	[1, 0]
]

expected = [
	[0],
	[0],
	[1],
	[1]
]

for i in range(10000) :
	for j in range(4) :
		NN.backward(examples[j], expected[j], 0.2)

