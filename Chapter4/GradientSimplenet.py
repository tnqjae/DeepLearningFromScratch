import sys, os
sys.path.append(os.pardir)

import numpy as np
from mbCrossEntropyError import mini_batch_cross_entropy_error

def softmax(a): #함수 명명 규칙에 따라 코드를 수정
    exp_a = np.exp(a) #지수 함수
    sumExpA = np.sum(exp_a)
    y = exp_a / sumExpA

    return y

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규 분포로 초기화
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(x)
        loss = mini_batch_cross_entropy_error(y, t)#차원의 불일치 때문에, 미니 배치를 사용하여 차원을 맞춰줘야 합니다.

        return loss
    

net = simpleNet() # 객체 생성 (발표 시에 객체에 대한 개념 가볍게 언급)
print(net.W)
print()

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

#p의 최댓값에 대한 인덱스
print("p의 최댓값에 대한 인덱스 : ", np.argmax(p))

t = np.array([0,0,1]) #정답 레이블
print("손실률 계산 : ", net.loss(x,t))

