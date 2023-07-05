import numpy as np

#수식을 통한 함수 구현
def softmax(a): #함수 명명 규칙에 따라 코드를 수정
    exp_a = np.exp(a) #지수 함수
    sumExpA = np.sum(exp_a)
    y = exp_a / sumExpA

    return y
