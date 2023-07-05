#mnist관련 예제 코드
import sys, os
sys.path.append(os.pardir)
from Chapter3.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten = True, normalize=False)
#normalize인수는 이미지 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화 할지 결정 False이면 0~255사이 값을 유지
#flatten는 이미지를 평탄하게, 즉 1차원 배열로 만들지

print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000, )
print(x_test.shape) #(10000, 784)
print(t_test) #(10000, )