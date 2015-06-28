#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: ann.py

"""
Developed With Python Version 2.7.8
"""

print __doc__

import numpy
import random

class NeuralNet:

	# Initialization
	def __init__ (self, data = None, targets = None, epochs = 10, etaDataLayer = 0.2, etaHiddenLayer = 0.02, hiddenLayers = 1, hiddenNeurons = 20, dataWeightsLimit = 0.05, hiddenWeightsLimit = 0.5, predictClasses = True, backPropagate = False, verbose = False):
		"""
		Initializes Neural Network With User Provided Parameters

		@param data 	: 
		"""
		self.data 				= {}
		self.data["input"] 		= data
		self.data["rows"] 		= None
		self.data["cols"] 		= None
		self.targets 			= targets 
		self.epochs 			= epochs
		self.etaDataLayer 		= etaDataLayer
		self.etaHiddenLayer 	= etaHiddenLayer
		self.hiddenLayers 		= hiddenLayers
		self.hiddenNeurons 		= hiddenNeurons
		self.dataWeightsLimit 	= dataWeightsLimit
		self.hiddenWeightsLimit	= hiddenWeightsLimit
		self.predictClasses 	= predictClasses
		self.backPropagate 		= backPropagate
		self.verbose 			= verbose
		if self.data["input"] != None:
			self.train(
				data 				= self.data["input"], 
				targets 			= self.targets, 
				epochs 				= self.epochs, 
				etaDataLayer 		= self.etaDataLayer,
				etaHiddenLayer 		= self.etaHiddenLayer,
				hiddenLayers 		= self.hiddenLayers,
				hiddenNeurons 		= self.hiddenNeurons,
				dataWeightsLimit  	= self.dataWeightsLimit,
				hiddenWeightsLimit 	= self.hiddenWeightsLimit,
				predictClasses 		= self.predictClasses,
				backPropagate 		= self.backPropagate,
				verbose 			= self.verbose
			)

		# Sigmoid Kernel Function
		def sigmoid (x):
			return 1 / (1 + numpy.exp(-x))


	# Train Neural Network
	def train (self, data, targets, epochs = 10, etaDataLayer = 0.2, etaHiddenLayer = 0.02, hiddenLayers = 1, hiddenNeurons = 20, dataWeightsLimit = 0.05, hiddenWeightsLimit = 0.5, predictClasses = True, backPropagate = False, verbose = False):
		self.data["input"] 		= numpy.array(data)
		self.data["rows"] 		= len(data)
		self.data["cols"] 		= len(data[0])
		self.targets 			= targets
		self.classCount 		= len(set(targets))
		self.epochs 			= epochs
		self.etaDataLayer 		= etaDataLayer
		self.etaHiddenLayer 	= etaHiddenLayer
		self.hiddenLayers 		= hiddenLayers
		self.hiddenNeurons 		= hiddenNeurons
		self.dataWeightsLimit 	= dataWeightsLimit
		self.hiddenWeightsLimit	= hiddenWeightsLimit
		self.predictClasses 	= predictClasses
		self.backPropagate 		= backPropagate
		self.verbose 			= verbose
		self.report 			= {}
		self.report["total"] 	= self.epochs * len(data)
		self.dataWeights 		= self.dataWeightsLimit * numpy.random.random_sample((self.data["cols"], self.hiddenNeurons))
		self.hiddenWeights 		= self.hiddenWeightsLimit * numpy.random.random_sample((self.hiddenNeurons + 1, 1))
		if self.verbose == True:
			print "------------------------------------"
			print "--- Training . . . -----------------"
			print "------------------------------------"
		for epoch in range(self.epochs):
			hits 				= 0
			misses 				= 0
			total 				= 0
			distances 			= []
			sampleIndices 		= sorted(range(len(self.data)), key = lambda k: random.random())
			for sampleIndex in sampleIndices:
				sample 			= self.data["input"][sampleIndex]
				target 			= self.targets[sampleIndex]
				if self.verbose == True:
					print "    Feeding Forward"
					print "         Sample", sampleIndex + 1, "of Epoch", epoch + 1
					print "         Known Class: ", target
				# Forward Propagation
				a 				= 1 / (1 + numpy.exp(- numpy.dot(sample, self.dataWeights)))
				b 				= numpy.concatenate([[1], a])
				output 			= 1 / (1 + numpy.exp(- numpy.dot(b, self.hiddenWeights)))
				# Metric Computation & Communication
				error 			= 0.5 * ((target - output) ** 2)
				distance 		= abs(target - output)
				if self.predictClasses == True:
					predictedClass 	= round(self.classCount * output)
				distances.append(distance)
				print output
				# # Multiple Hidden Layer Routine
				# for hiddenLayer in range(hiddenLayers):



			# print randomizedSample
		if self.verbose == True:
			print "------------------------------------"
			print "--- Training Complete --------------"
			print "------------------------------------"

if __name__ == "__main__":
	data 		= [
		[0, 1, 0, 0, 0, 0],
		[1, 0, 1, 0, 0, 0],
		[0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 1]
	]
	classes 	= [1, 0, 0, 1]
	ann 		= NeuralNet()
	ann.train(data, classes, verbose = True)
	# ann.init(data, classes)
	# ann.train(data, classes)