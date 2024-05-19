https://www.youtube.com/watch?v=lp1C0P6mBfw&list=PLhhyoLH6IjfzqE1Z9uGrTb66tcu311C7c

## Part 1: Standard Notation 

| Input | Hidden | Output |
| ----- | ------ | ------ |
|       | ()     |        |
| ()   | ()     | ()     |
| ()    | ()     | ()     |
|  | () |  |

Input = layer 0
Hidden = layer (1-x)
Ouput = layer (x)


* `A` = *output from any layer*
    * $A_j$ (j = node)
    * $A^l$ (l = layer)
        * $A^2_1$ = output for layer 2 node 1
    * $A^i$ (i = training example)
        * $A^1_(2,3)$ = output for layer 2 node 3 at training example 1
* `w` = weight = *value modifications, or edge, between layers*
    * $w^[l]$ (l = layer)
    * $w_(j,k)$ (node start j, node end k)
        * $w^[1]_(2,3)$ = weight going INTO node 1 connecting second node in node 0 to third node in node 1

- Each node is a summation: $$A^{[l+1]} = \sum_{j=1}^{n^{[l]}} A^{[l]}_j \cdot w^{[l]}_{j,i} + b$$

- bias  term $b^[1]_1$ = local to specific node (add y-intercept)
    literally linear function y = mx + b

- relu, tanh, sigmoid = nonlinear activation function (forcing function)
    A^1_1 = relu(z)

===============================================
## Part 2: Forward Propagation
Matrix form: 

* $A^{[0]}$ = (examples = handwritten images digits, features (aka pixels) in previous layer)
* $w^{[l]}$ = (number of features in prev layer, number of features in next layer)
* $z^{[1]}$ = $A^{[0]}$ * $w^{[1]}$ + $b^{[1]}$
    * matrix multiplication: features from previous layer cancel out!
    * $z^{[1]}$= (examples, features of layer 1) + (1,features of layer 1)
    * addition = *broadcasting* (expand into rows; same quantity as examples)
* $A^{[1]}$ = relu($z^{[1]}$)
    * element-wise for each node and for all training examples
    * relu = max(0,z^[1])
* z^[2] = A^[1]*w^[2] + b^[2] = *activation*
* output layer ALWAYS has softmax: turn scores of z to probabilities
    * f_j(z) = e^z_j / sum(e^[z_k])
    * = 0 or 1 (normalized them; value of one value / sum of all values )
        * probabilities = softmax(z^[2]) = all training examples and all the nodes
* loss: L_[i] = -log(f_y_i(z))
    * general case for z
    * f = computed from neural network
    * y_i = correct label for training example
    * L_i = -log((e^z_correct label) / sum of all classes for e^[z_c])

===============================================
## Part 3: Back Propagation
 
A^[0] --> z^[1] as a function of w^[1] and b^[1] --> A^[1] ... --> Loss
    Goal: compute loss, then change weights and bias terms to get better gradients
L_i = -log((e^z_correct label) / sum of all classes for e^[z_c])
     = -log(e^z_yi) + log(sum of all classes e^[z_c])
    
deriv of loss wrt z^[2]_k (output layer's gradient) = (d / dz^[2]_k) * (-z_y_i) + (d / dz^[2]_k)*(log(sum for e^z_c))
    = -1 [0 if y_i = k] (aka 0 if not correct) + 1/sum(e^z_c) *d/dz^[2]_k * sum for e^z[2]_c
        partial deriv of a summation = 1 bc partial deriv is for a specific activation and all others are = 0
    = e^z_k / sum for e^z_c -1 [0 if y_i = k]
Solve for weights now: 
    take deriv z2 wrt deriv weight

dz^[2] / dw^[2] = d/dw^[2] * (A^[1]*w^[2] + b^[2]) = A^[1]

for the biases: 

dz^[2] / db^[2] = d/db^[2] * (A^[1]*w^[2] + b^[2]) = I = 1

when update biases, b^[2] - dz^[2] / db^[2]
   size(1,features)  - size(example, features in layer 2)
        when having diff sizes, add gradients together 

keep moving backwards for backprop: 

dz^[2] / dA^[1] = d / dA^[1] * (A^[1]*w^[2] + b^[2]) = (w^[2])T

dz^[1] / dA^[1] =  d / dA^[1] (max(0,z^[1]))
        NB: when taking derivative of 1 x n matrix, dF / dA = n x n bc Jacobian form




