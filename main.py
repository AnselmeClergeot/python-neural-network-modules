#coding: utf-8

from Neural_Network import *

NN = Neural_Network(2, 2, 1)

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
		NN.backward(t[j], e[j], 0.2)

for ex in t :
	print(ex)
	print(NN.forward(ex))
