import numpy as np
from GradientEx import numerical_gradient #기울기 계산 함수를 import

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    #f는 최적화하려는 함수, init_x는 x의 초깃값, lr -> learing rate로 학습률, step_num은 경사법의 반복 횟수
    x = init_x #x값 초기화

    for i in range(step_num):
        grad = numerical_gradient(f, x)# 함수의 기울기를 구하기
        x -= lr * grad # 학습률을 곱한 값으로 갱신
        
    
    return x


