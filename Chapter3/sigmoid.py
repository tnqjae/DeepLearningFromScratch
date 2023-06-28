import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #시그모이드 함수 수식을 통해 구현 exp는 자연상수

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

#시그모이드 함수와 계단 함수의 차이점은 매끄러움의 차이
#매끄러움은 신경망 학습에서 중요한 역할
