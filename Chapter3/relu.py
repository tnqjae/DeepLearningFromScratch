import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x) #두 값 중 큰 값을 반환하는 함수


x = np.arange(-6.0, 6.0, 1)
y = relu(x)

plt.plot(x,y)
plt.ylim(-1.1, 5.5)
plt.show()