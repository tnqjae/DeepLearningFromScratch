import sys, os
sys.path.append(os.pardir)
from GradientDescent import numberical_gradient
from CrossEntropyError import  cross_entropy_error
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #시그모이드 함수 수식을 통해 구현 exp는 자연상수

def softmax(a): #함수 명명 규칙에 따라 코드를 수정
    exp_a = np.exp(a) #지수 함수
    sumExpA = np.sum(exp_a)
    y = exp_a / sumExpA

    return y

class TwoLayerNet:
    #2층짜리 신경망의 학습 알고리즘 구현
    #초기화(입력층의 뉴런 수, 은닉층의 뉴런 수, 출력층의 뉴런 수)
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
        
        self.params = {} #파라미터를 담는 변수 선언
        
        #가중치와 편향
        #가중치와 편향의 설정은 신경망 학습에서 매우 중요, 하지만 이번 장에서는 알아보지 않고 가중치는 정규분포에 따른 난수로, 편향은 0으로 초기화

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #0.01 * input_size부터 hidden_size사이의 난수 값 1개
        self.params['b1'] = np.zeros(hidden_size) #hidden_size와 같은 크기로 배열을 생성하고 그 값을 전부 0으로 초기화
        
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    #예측(추론)을 수행, x는 이미지 데이터
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    #정확도를 구하는 메서드
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy

    #손실 함수 값을 구함, x는 이미지 데이터, t는 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    #수치 미분으로 기울기를 구함, 시간이 오래 걸림
    #고속으로 수행하고 싶다면 오차역전파를 이용해서 계산해야함.
    def numberical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numberical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numberical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numberical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numberical_gradient(loss_W, self.params['b2'])

        return grads
    
#예시
net = TwoLayerNet(input_size= 784, hidden_size= 100, output_size= 10)

net.params['W1'].shape #(784, 100)
net.params['b1'].shape #(100, )
net.params['W2'].shape #(100,10)
net.params['b2'].shape #(10, )

x = np.random.rand(100, 784)
y = net.predict(x)
"""
grads변수에는 params변수에 대응하는 각 매개변수의 기울기가 저장됨
nuberical_gradient 메서드를 이용해 기울기를 계산하면 grads에 기울기 값이 저장
"""
