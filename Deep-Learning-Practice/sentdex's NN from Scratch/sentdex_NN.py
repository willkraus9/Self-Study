#adapted from https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=1
#adapted from https://cs231n.github.io/neural-networks-case-study/

import numpy as np
# import spiral_data
import matplotlib.pyplot as plt

np.random.seed(0)

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1* np.random.randn(n_inputs,n_neurons) #get in range btwn 0 and 1; already transpose weights (should be neuron x inputs)
        self.biases = np.zeros((1,n_neurons)) # if going to zero, can change to non-zero numbers for initial weights + biases
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+ self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
        
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis= 1, keepdims=True)) #subtract largest value, keeps between 0 and 1 for e^x
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims= True) # key to softmax
        self.output = probabilities

class Loss:
    def calculate(self,output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7, 1-1e-7)
        if len(y_true.shape) ==1: #one hot encoding: classification is in this form [[0,1][1,0]]
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2: #two hot encoding: [0,1,1,0]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
        
        
#Running forward pass
X,y = spiral_data(100,3)

dense1=Layer_Dense(2,3)
activation1=Activation_ReLU()

dense2=Layer_Dense(3,3)
activation2=Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])

loss_function = Loss_Categorical_Cross_Entropy()
loss = loss_function.calculate(activation2.output, y)
print(loss)

#Categorical cross-entropy: L = -sum_j (y_i,j * log(yhat_i,j)) = -log(yhat_i,k)
    # fix log(0) issue: clip to 1e-7

#Plotting
X,y = spiral_data(100,3)

plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c = y, cmap = "brg")
plt.show()

# layer1 = Layer_Dense(4,5)
# activation1 = Activation_ReLU()

# layer1.forward(X)
# print(layer1.output)
# activation1.forward(layer1.output)
# print(activation1.output)

# print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)    


# weights = [[0.2,0.8,-0.5,1.0],
#            [0.5,-0.91,0.26,-0.5],
#            [-0.26,-0.27,0.17,0.87]]

# biases = [2,3,0.5]

# weights2 = [[0.1,-0.14,0.5],
#            [-0.5,0.12,-0.33],
#            [-0.44,0.73,-0.13]]

# biases2 = [-1,2,-0.5]
# # print(np.array(inputs).T.shape)
# # print(np.array(weights).shape)
# layer1_outputs = np.dot(np.array(inputs), np.array(weights).T) + biases
# layer2_outputs = np.dot(np.array(layer1_outputs), np.array(weights2).T) + biases2

# print(layer2_outputs)

# #in general, 32 < batch_size < 128 

# # layer_outputs = []
# # for neuron_weights, neuron_bias in zip(weights,biases):
# #     neuron_output = 0
# #     for n_input,weight in zip(inputs,neuron_weights):
# #         neuron_output+=n_input*weight
# #     neuron_output+=neuron_bias
# #     layer_outputs.append(neuron_output)

# # print(layer_outputs)

