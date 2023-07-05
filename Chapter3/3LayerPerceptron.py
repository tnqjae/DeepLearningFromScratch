import numpy as np

#Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Input Layer to Layer 1
X = np.array([1.0, 0.5])
W1 = np. array([0.1, 0.3, 0.5], [0.2, 0.4, 0.6])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

#Sum of weights (sum of weighted signals and biases)
Z1 = sigmoid(A1)
print(A1)
print(Z1)

#Signal transmission from the 1st Layer to the 2nd Layer
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape) # (3,)
print(W2.shape) # (3, 2)
print(B2.shape) # (2, )

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

#Signal transmission from the 2nd Layer to the Output Layer
def identity_function(x):
    return 0 

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([[0.1, 0.2]])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)