''' simple_BPNN.py an obect-oriented framework for multilayer backpropagation neural network'''
__author__  = "Daniel C. Elton"
__maintainer__ = "Daniel C. Elton"
__copyright__ = "Copyright 2016, Daniel C. Elton"
__license__ = "MIT"
__status__ = "Development"

''' I used http://deeplearning.net/tutorial/mlp.html for a bit of inspiration'''

import numpy as np, pandas as pd

#---------------------------------------------------------------
def act_fn(x):
	'''the activation function (currently logistic function)
	   input :  
			x: a nx1 numpy ndarray
	   return:  
			a nx1 numpy ndarray
	'''
	bias = -.5
	x = x + bias
	act = 1/(1 + np.exp(-x)) 
	return act 

#---------------------------------------------------------------
def der_act_fn(x):
	'''the derivative of the activation function
	   input :  
			a_vec: a nx1 numpy ndarray
	   return:  
			a nx1 numpy ndarray
	'''
	bias = -.5
	x = x + bias
	der = x.dot(np.exp(-x)/np.square(1 + np.exp(-x)))
	return der 


#---------------------------------------------------------------
def one_hot(labels, n_classes):
	'''given a vector of n numerical labels, it spits out an array of dimension 
	num_labels X n_classes. Each row contains a 'one hot' vector. 
	   ie. for 10 classes labeled by the digits 1-10: 
	   1 ->  1 0 0 0 0 0 0 0
	   2 ->  0 1 0 0 0 0 0 0
	   ...
	   10 -> 0 0 0 0 0 0 0 1
	 '''
	num_labels=len(labels)

	one_hot_labels = np.zeros([n_classes,num_labels])

	for i in xrange(num_labels):
		j = labels[i]
		one_hot_labels[j,i] = 1

	return one_hot_labels

#---------------------------------------------------------------
class layer(object):
	'''layer objects contains several important attributes: 
		self.weights - a ndarray of dimension (n_in x n_nodes)
		self.wsum    - the weighted sum of inputs into the layer
		self.deltas  - the deltas for this layer
		self.activations - the activations of the layer
	'''
	def __init__(self, n_in, n_nodes, act_fun='sigmoid'):
		#random initalization of weights for this layer 
		if (act_fun=='sigmoid'):
			scale_fac = 4*np.sqrt(6)/(np.sqrt(n_in+n_nodes))
			self.weights = scale_fac*(2*np.random.rand(n_nodes,n_in)-1)
		self.size   = n_nodes
		self.wsum   = np.zeros(n_nodes)
		self.deltas = np.zeros(n_nodes)

#	def get_weights(self):
#		return self.weights

#	def set_weights(self, weights):
#		self.weights = weights

#	self.weights = property(get_weights, set_weights, doc="a weight matrix")
   
	def activate(self,inp):
		self.wsum = self.weights.dot(inp)
		return act_fn(self.wsum)



#---------------------------------------------------------------
class neural_net(object):
	'''A neural_net object is simply a list of layers'''
	
	#-------------------------------------------
	def __init__(self, n_in, nodes_per_layer):
		self.n_layers = len(nodes_per_layer)
		self.ni = n_in
		self.no = nodes_per_layer[self.n_layers-1]
		self.layer_sizes=nodes_per_layer
		self.layers = []
		
		#create first layer (layer 0)
		self.layers += [layer(n_in, nodes_per_layer[0])]
		
		#create hidden layers and output layer
		for i in range(1, self.n_layers):
			self.layers += [layer(nodes_per_layer[i-1],nodes_per_layer[i])]
		
	#-------------------------------------------
	def activate(self, vec):
		'''activate each layer of the neural network
			output:
			vec - an ndarray of size (n_out x 1) representing the activation of the last layer
		'''
		for i in range(self.n_layers):
			vec = self.layers[i].activate(vec)
		return vec
	
	#-------------------------------------------
	def backpropagation(self, activation, target, learn_rate):
		'''perform backpropagation on the network'''
		n_layers = self.n_layers
		layers = self.layers
		
		#calculate deltas for last layer
		layers[n_layers-1].deltas = der_act_fn(layers[n_layers-1].wsum)*(activation - target)

		#calculate deltas for hidden layers
		for l in range(n_layers-2, 0, -1):
			layers[l].deltas = der_act_fn(layers[l].wsum)*( np.transpose(layers[l+1].weights))*layers[l+1].deltas
	
		#update weights for last and hidden layers
		for l in range(n_layers-1, 1, -1):
			for i in range(layers[i].size):
				for j in range(layers[i-1].size):
					layers[l].weights[i,j] = layers[l].weights[i,j] + learn_rate*layers[l].deltas[i]*layers[i-1].wsum[j]
	
		#update weights for first (zeroeth) layer
		for i in range(layers[0].size):
			for j in range(len(target)):
				layers[0].weights[i,j] = layers[0].weights[i,j] + learn_rate*layers[0].deltas[i]*target[j]


	#-------------------------------------------
	def training_epoch(self, data, targets, learning_rate):
		'''Input:
				data : a batch of training data. type ndarray of size (n_inputs x n_examples)
				learning_rate
			Output: 
		'''
		n_examples = np.size(data,1)
		
		avg_cost = 0
		for i in range(n_examples):
			activation = self.activate(data[:,i])
			print activation - targets[:,i]
			avg_cost += np.square(activation - targets[:,i])
			self.backpropagation(activation, targets[:,i], learning_rate)
		avg_cost = avg_cost/n_examples
		return avg_cost

	#-------------------------------------------
	def regularization(self):
		'''return a regularization term for the neural network'''
		reg = 0 
		for layer in self.layers:
			reg += sum(layer.weights**2)
		return reg
	
	#-------------------------------------------
	def cost(self, data):
		'''given a set of input data, calculates the cost, or error outputed by the nn
			this method is provided for use with advanced optimization routines
		'''
		num_inps = np.size(data,1)
	
		cost=0
	
		for i in range(num_inp):
			outvec = self(data[:,i])
			cost = cost + sum( ans[:,i]*np.log(outvec) + (1-ans[:,i])*np.log(1-outvec) )
			cost = cost + .1*nn.regularization()
			return cost
	


#----------------------------------------------------------------
#------------------- Setting up and training -------------------
#----------------------------------------------------------------
def train(n_epochs = 10,
		  learning_rate=.1,
		  datafile='train.csv',
		  nodes_per_layer=[5,5],
		  regularization=True,
		  reg_param=.01):
	#import data
	#raw_data = pd.read_csv(datafile, delimiter=",").values
	
	#xor test input
	raw_data = np.array([[0,0,0],[1,0,1],[1,1,0],[0,1,1]])  
		
 	#create array of target vectors (stored in columns)
	one_hot_labels = one_hot(labels=raw_data[:,0], n_classes=2)
	
	#data (examples stored in columns)
	data = np.transpose(raw_data[:,1:])
	
	#build neural net
	nn = neural_net(n_in=2, nodes_per_layer=[5,5,2])

	for n in range(n_epochs):
		avg_cost = nn.training_epoch(data, data, learning_rate)
		print avg_cost



if __name__ == '__main__':
	train()
