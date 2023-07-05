import pickle
import numpy as np
import sys, os
sys.path.append(os.pardir)
from Chapter3.mnist import load_mnist #mnist.py에서 load_mnist 함수를 불러옴
from Chapter3.softmax import softmax

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #시그모이드 함수 수식을 통해 구현 exp는 자연상수

#신경망의 추론 처리를 위한 함수들

def get_data():
    #데이터를 불러오는 함수
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    
    return x_test, t_test

def init_network():
    #가중치 샘플을 불러옴
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    
    return network

def predict(network, x):
    #입력 이미지를 받아 예측값을 반환

    W1, W2, W3 = network['W1'], network['W2'], network['W3'] # Weight
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] # Bais

    #첫 번째 은닉층
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    #두 번째 은닉층
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    #출력층
    a3 = np.dot(z2, W3) + b3
    
    #출력값을 확률로 변환
    y = softmax(a3)
    
    return y

#추론을 수행 후, 정확도 평가
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    #x에 저장되어 있는 이미지 데이터를 1장씩 꺼내 predict 함수로 분류
    y = predict(network, x[i])
    p = np.argmax(y) #확률이 가장 높은 원소의 인덱스를 얻음
    #정답 레이블과 비교하여 맞으면 정답 결과를 + 1
    if p == t[i]:
        accuracy_cnt += 1

#정답 결과와 전체 데이터 개수를 나누어 정확도를 계산
print("Accuracy : " + str(float(accuracy_cnt) / len(x)))

