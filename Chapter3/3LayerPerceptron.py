import numpy as np

#활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#입력층에서 1층으로 신호전달
X = np.array(1.0, 0.5)
W1 = np. array([0.1, 0.3, 0.5], [0.2, 0.4, 0.6])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

#가중치의 합(가중 신호와 편향의 총합
Z1 = sigmoid(A1)
print(A1)
print(Z1)
