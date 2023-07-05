import numpy as np

#수식을 통한 함수 구현
def SoftMax(a):
    exp_a = np.exp(a) #지수 함수
    sumExpA = np.sum(exp_a)
    y = exp_a / sumExpA

    return y
