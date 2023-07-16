import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from TwoLayerNet import TwoLayerNet

#훈련 이미지, 훈련 레이블 / 시험 이미지, 시험 레이블
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)\

train_loss_list = []

#하이퍼 파라미터
iters_num = 10000 #반복 횟수
train_size = x_train.shape[0]
print(train_size)
batch_size = 100 #미니배치 크기(무작위로 몇 개의 데이터를 꺼낼 것인가)
learning_rate = 0.1

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

#그래프를 그리기 위한 코드
train_loss_list = []

for i in range(iters_num):
    #미니 배치 획득
    batch_mask = np.random.choice(train_size, batch_size,1)
    x_batch = x_train[batch_mask] #훈련 이미지에 대한 배치
    t_batch = t_train[batch_mask] #훈련 레이블에 대한 배치


    #기울기 계산
    grad =  network.gradient(x_batch, t_batch) #훈련 이미지와 레이블에서 추출한 배치를 바탕으로 기울기를 구함

    #매개변수 갱신(중요한 부분)
    for key in ('W1', 'b1', 'W2', 'b2'):
        #경사 하강법은 매개변수 값을 줄여 나가면서 손실함수를 최소화(최적화)하는 기법, 그래서 -=을 해준다.
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

#손실함수 변화 추이 그래프
plt.plot(train_loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Change in Loss Function Value")
plt.show()