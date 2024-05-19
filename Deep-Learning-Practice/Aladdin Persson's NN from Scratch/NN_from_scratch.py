# see https://www.youtube.com/watch?v=NJvojeoTnNM&list=PLhhyoLH6IjfzqE1Z9uGrTb66tcu311C7c&index=4 for reference
import numpy as np
from utils import create_dataset, plot_contour 

class NeuralNetwork():
    def __init__ (self, X, y):
        # m training ex, n features
        self.m, self.n = X.shape
        # self.lambd = 1e-3 # regularization parameter
        self.learning_rate = 0.1
        
        #size of NN:
        self.h1 = 25 
        self.h2 = len(np.unique(y)) # number classes
    
    def init_kaiming_weights(self, l0, l1):
        # nodes in previous layer, nodes in next layer
        # could also do normal distribution, but this one apparanetly works better
        w = np.random.randn(l0,l1) * np.sqrt(2.0 / l0)# number of noedes coming from, number of nodes going to 
        b = np.zeros((1,l1))
        return w, b
    
    def forward_prop(self, X, parameters):
        # params = cache dictionary; store w,b in parameters
        W2= parameters["W2"]
        W1= parameters["W1"]
        b2= parameters["b2"]
        b1= parameters["b1"]
        
        a0 = X
        z1 = np.dot(a0,W1) + b1
        a1 = np.maximum(0,z1)
        z2 = np.dot(a1, W2) + b2
        
        # softmax for output
        scores = z2 
        exp_scores = np.exp(scores)
        # number of training examples , number of classes
        
        probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
        # each node is probability that it is a certain class
        
        cache = {"a0" : X, "probs" : probs, "a1" : a1}
        
        return cache, probs
          
    def compute_cost(self, y, probs, parameters):
        W2 = parameters["W2"]
        W1 = parameters["W1"]
        y = y.astype(int)

        # regularization and data loss
        data_loss = np.sum(-np.log(probs[np.arange(self.m), y]) / self.m)
            # CEM: take log only of correct class
        reg_loss = 0.5 * 1e-3 * np.sum(np.square(W1)) + 0.5*1e-3*np.sum(np.square(W2))
        total_cost = data_loss + reg_loss
        
        return total_cost
            
    def back_prop(self, cache, parameters, y):
        W2 = parameters["W2"]
        W1 = parameters["W1"]
        
        a0 = cache["a0"]
        a1 = cache["a1"]
        probs= cache["probs"]
        
        #want dW1, db1, dW2, db2
        dz2 = probs
        dz2[np.arange(self.m), y] -= 1
        dz2 /= self.m
        
        #backprop for dW2,db2
        dW2 = np.dot(a1.T, dz2) + 1e-3 * W2
        db2 = np.sum(dz2, axis = 0, keepdims = True)
        
        #backprop for dW1, db1
        dz1 = np.dot(dz2, W2.T)
        dz1 = dz1 * (a1 > 0) # relu
        
        dW1 = np.dot(a0.T, dz1) + 1e-3 * W1
        db1 = np.sum(dz1, axis = 0, keepdims = True)
        
        grads = {"dW1" : dW1,
                 "db1" : db1,
                 "dW2" : dW2,
                 "db2" : db2}
        
        return grads
        
    def update_parameters(self, parameters, grads):
        #grad descent step
        
        W2 = parameters["W2"]
        W1 = parameters["W1"]
        b2 = parameters["b2"]
        b1 = parameters["b1"]
        
        dW2 = grads["dW2"]
        dW1 = grads["dW1"]
        db2 = grads["db2"]
        db1 = grads["db1"]
        
        W2 -= self.learning_rate * dW2
        W1 -= self.learning_rate * dW1
        b2 -= self.learning_rate * db2
        b1 -= self.learning_rate * db1
        
        parameters = {"W1": W1, "W2": W2, "b1":b1, "b2":b2}
        return parameters
    
    def main(self, X, y, num_iters = 10000):
        W1,b1 = self.init_kaiming_weights(self.n, self.h1)
        W2,b2 = self.init_kaiming_weights(self.h1, self.h2)

        parameters = {"W1": W1, "W2": W2, "b1" : b1, "b2":b2}
    
        for i in range(num_iters+1):
            # forward prop
            cache,probs = self.forward_prop(X,parameters)
            cost = self.compute_cost(y,probs,parameters)
            if i % 2500 ==0:
                print(f"At iteration {i}, we have a loss of {cost}")
            grads = self.back_prop(cache, parameters, y)
            
            parameters = self.update_parameters(parameters, grads)
            
        return parameters      
            
if __name__ == "__main__":
    X, y = create_dataset(N = 300, K=3)
    y = y.astype(int)
    
    # Train network
    NN = NeuralNetwork(X, y)
    trained_parameters = NN.main(X, y)

    # Get trained parameters
    W2 = trained_parameters["W2"]
    W1 = trained_parameters["W1"]
    b2 = trained_parameters["b2"]
    b1 = trained_parameters["b1"]

    # Plot the decision boundary 
    plot_contour(X, y, NN, trained_parameters)