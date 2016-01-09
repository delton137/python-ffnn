#Simple Backpropagation Neural Network using NumPy

Motivation: 
The motivation for this was my desire to 'fiddle around' with a neural network. There were many things I want to fiddle with: 
* number of hidden layers & number of nodes in each layer 
* learning rate
* the activation function (nonlinear vs linear, etc)
* tweaking bias, input scaling, etc 
* the introduction of momentum to the backpropagation of deltas
* 'online' vs batch learing
* using advanced optimization instead of backprop
* modification of the cost function for adv. opt. including degree and type of regularization 


Now I do not consider Python ideal for neural networks, because it is often slow. However, Python is fun for fooling around. 

I encountered two problems, however. Most of the simple python codes implementing backpropagation only contain 1 hidden layer. I wanted to experiment 
with an arbitrary number of hidden layers, each of arbitrary size.

Packages like PyBrain allow you do this easily, and are great for learning. However, I figured in the time 
I figured out all that I wanted to do with PyBrain, I could have implemented my own sytem, and probably learned more in the process.
This project has made me appreciate some of the difficulties of coding a neural network. 
I have a inbuilt desire to vectorize everything for speed, which I had to resist in favor of object-oriented readibility. 

