#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: ann.py

"""
Developed With Python Version 2.7.8
"""

print __doc__

import numpy
import random

class ANN:

	# Initialization
	def __init__ (self, data = None, targets = None, epochs = 10, etaDataLayer = 0.2, etaHiddenLayer = 0.02, hiddenLayers = 1, hiddenNeurons = 20, dataWeightsLimit = 0.05, hiddenWeightsLimit = 0.5, regression = True, backPropagate = False, verbose = False):
		"""
		Initializes Neural Network With User Provided Parameters – If Data is Provided, Automatically Initializes Training

		@params: type - description
			data 				: 2x2 numpy arrays or python lists 	- Samples by Features
			targets 			: Numpy array or python list 		- Target classes or values for supervised learning
			epochs 				: Integer 							- Number of training epochs
			etaDataLayer 		: Float 							- Learning rate for first weights layer
			etaHiddenLayer 		: Float 							- Learning rate for hidden layer(s)
			hiddenLayers 		: Integer 							- Number of hidden layers
			hiddenNeurons 		: Integer 							- Number of neurons per hidden layer
			dataWeightsLimit 	: Float 							- First layer weights initialization limit around 0 (the range is effectively double the input value)
			hiddenWeightsLimit 	: Float 							- Hidden layer weights initialization limit around 0 (the range is effectively double the input value)
			regression 			: Boolean 							- If False, run classification mode
			backPropagate 		: Boolean 							- Whether to use backpropagation algorithm
			verbose 			: Boolean 							- Verbose output of training progress
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
		self.regression 		= regression
		self.backPropagate 		= backPropagate
		self.verbose 			= verbose
		self.report 			= {}
		self.dataWeights 		= None
		self.hiddenWeights 		= None
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
				regression 			= self.regression,
				backPropagate 		= self.backPropagate,
				verbose 			= self.verbose
			)

		# Sigmoid Kernel Function
		def sigmoid (x):
			return 1 / (1 + numpy.exp(-x))


	# Train Neural Network
	def train (self, data, targets, epochs = 100, etaDataLayer = 0.2, etaHiddenLayer = 0.2, hiddenLayers = 1, hiddenNeurons = 200, dataWeightsLimit = 0.05, hiddenWeightsLimit = 0.5, regression = True, backPropagate = False, verbose = False):
		"""
		Train Neural Network With User Provided Parameters

		@params: type - description
			data 				: 2x2 numpy arrays or python lists 	- Samples by Features
			targets 			: Numpy array or python list 		- Target classes or values for supervised learning
			epochs 				: Integer 							- Number of training epochs
			etaDataLayer 		: Float 							- Learning rate for first weights layer
			etaHiddenLayer 		: Float 		3					- Learning rate for hidden layer(s)
			hiddenLayers 		: Integer 							- Number of hidden layers
			hiddenNeurons 		: Integer 							- Number of neurons per hidden layer
			dataWeightsLimit 	: Float 							- First layer weights initialization limit around 0 (the range is effectively double the input value)
			hiddenWeightsLimit 	: Float 							- Hidden layer weights initialization limit around 0 (the range is effectively double the input value)
			regression 			: Boolean 							- If False, run classification mode
			backPropagate 		: Boolean 							- Whether to use backpropagation algorithm
			verbose 			: Boolean 							- Verbose output of training progress
		"""
		self.data["input"] 		= numpy.array(data)
		self.data["rows"] 		= len(data)
		self.data["cols"] 		= len(data[0])
		self.targets 			= targets
		self.classCount 		= len(set(targets))
		self.epochs 			= epochs
		self.etaDataLayer 		= float(etaDataLayer)
		self.etaHiddenLayer 	= float(etaHiddenLayer)
		self.hiddenLayers 		= hiddenLayers
		self.hiddenNeurons 		= hiddenNeurons
		self.dataWeightsLimit 	= float(dataWeightsLimit)
		self.hiddenWeightsLimit	= float(hiddenWeightsLimit)
		self.regression 		= regression
		self.backPropagate 		= backPropagate
		self.verbose 			= verbose
		self.report 			= {}
		self.report["total"] 	= self.epochs * len(data)
		self.dataWeights 		= self.dataWeightsLimit * numpy.random.random_sample((self.data["cols"], self.hiddenNeurons))
		self.hiddenWeights 		= self.hiddenWeightsLimit * numpy.random.random_sample((self.hiddenNeurons + 1, 1))
		print "========================================"
		print "--- Training . . . ---------------------"
		print "========================================"
		for epoch in range(self.epochs):
			hits 				= 0
			misses 				= 0
			processed 			= 0
			distances 			= []
			predictedClass 		= None
			sampleIndices 		= sorted(range(len(self.data["input"])), key = lambda k: random.random())
			if self.verbose == True:
				print "----------------------------------------"
				print "--- Epoch", epoch
				print "----------------------------------------"
			for sampleIndex in sampleIndices:
				print self.dataWeights
				sample 			= self.data["input"][sampleIndex]
				target 			= self.targets[sampleIndex]
				if self.verbose == True:
					print "--- Feeding Forward . . . --------------"
					print "         Sample", sampleIndex + 1, "of Epoch", epoch + 1
				# Forward Propagation
				a 				= 1.0 / (1.0 + numpy.exp(- numpy.dot(sample, self.dataWeights)))
				b 				= numpy.concatenate([[1], a])
				output 			= 1.0 / (1.0 + numpy.exp(- numpy.dot(b, self.hiddenWeights)))[0]
				# Metric Computation & Communication
				if self.regression == False:
					error 			= 0.5 * (((target / (self.classCount - 1)) - output) ** 2)
					distance 		= abs(target - (output * (self.classCount - 1)))
					predictedClass 	= round(self.classCount * output) - 1
					if predictedClass == target:
						hits += 1
					else:
						misses += 1
					if self.verbose == True:
						print "         Annotated Class: \t", target
						print "         Computed Class: \t", predictedClass
						print "         Computed SSE: \t\t%0.4f" % error
						print "         Raw Distance: \t\t%0.4f" % distance
						if predictedClass == target:
							print "         Prediction Status: \tHit! :)"
						else:
							print "         Prediction Status: \tOops! :("

				else:
					error 			= 0.5 * ((target - output) ** 2)
					distance 		= abs(target - output)
					if self.verbose == True:
						print "         Annotated Value: \t", target
						print "         Computed Value: \t%0.4f" % output
						print "         Computed SSE: \t\t%0.4f" % error
						print "         Raw Distance: \t\t%0.4f" % distance
				processed += 1
				distances.append(distance)
				# Back Propagation
				if self.verbose == True:
					print "--- Back Propagating . . . -------------"
				deltaDataWeights 	= ((target  / self.classCount - 1) - output) * output * (1 - output)
				deltaHiddenWeights 	= numpy.delete((b * (1 - b) * self.hiddenWeights * deltaDataWeights)[0], 0.0)
				updateHiddenWeights = etaHiddenLayer * b * deltaDataWeights
				updatedHiddenLayer 	= b + updateHiddenWeights
				self.hiddenWeights 	= numpy.transpose(numpy.atleast_2d(updatedHiddenLayer))
				updateDataWeights 	= etaDataLayer * numpy.outer(sample, deltaHiddenWeights)
				self.dataWeights  	= self.dataWeights + updateDataWeights
				print self.dataWeights
				if self.verbose == True:
					print "--- Sample Processed -------------------\n"

				# updateDataWeights 	= etaDataLayer * numpy.array(sample) * deltaHiddenWeights

				# # Multiple Hidden Layer Routine
				# for hiddenLayer in range(hiddenLayers):
			if self.verbose == True:
				print "----------------------------------------"
				print "--- Epoch", epoch, "Complete"
				print "----------------------------------------"
				if self.regression == False:
					accuracy 		= (hits/processed) * 100
					print " 	Epoch Hits / Total:\t", hits, "/", processed
					print " 	Epoch Hit Percent:\t%0.2f" % (float(hits) / processed * 100), "\n"
				else:
					print "\n"
		print "========================================"
		print "--- Training Complete ------------------"
		print "========================================"

if __name__ == "__main__":
	ann 		= ANN()
	ann.train(data, classes, epochs = 2, verbose = True, regression = False)
else:
	pass