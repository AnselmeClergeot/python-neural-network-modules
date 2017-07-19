#coding: utf-8

from Neural_Network import *
from mnist import MNIST

def fromClassToOutput(classValue) :
	output = [0] * 10
	output[classValue] = 1
	return output

def fromOutputToClass(output) :
	return output.argmax()

imageWidth = 28
numberOfClasses = 10

print("Creating neural network...")
NN = Neural_Network(imageWidth**2, 80, numberOfClasses)
NN.loadFromFile("networkConfig")

print("Done.")
print("Loading dataset...")

database = MNIST("database")
images, expected = database.load_training()

print("Done.")
print("Starting learning...")

for image in images :
	print(database.display(image))
	print("Network thinks it's a {}.".format(fromOutputToClass(NN.forward([normalize(x, 255) for x in image]))))
	raw_input()

for epoch in range(100) :
	for i in range(len(images)) :
		normalizedImage = [normalize(greyScale, 255) for greyScale in images[i]]
		NN.backward(normalizedImage, fromClassToOutput(expected[i]), 0.2)

		if i != 0 :
			if i % 100 == 0 :
				print("Epoch {} : {} images learned.".format(epoch, i))
		if i % 5000 == 0 :
			NN.saveToFile("networkConfig")
			print("Network configuration saved !")
