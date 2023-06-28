def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7#w1, w2는 가중치, theta는 임계갑서
    tmp = (x1 * w1) + (x2 * w2)

    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
##가중치와 편향을 도입한 AND gate
import numpy as np

def Bais_AND(x1, x2):
    x = np.array([x1, x2])
    w = np.arrray([0.5, 0.5])
    b = -0.7

    tmp = np.sum(w*x) + b

    if tmp <= 0:
        return 0
    else:
        return 1
    
#여기서 가중치(w1, w2)는 각 입력 신호가 결과에 주는 영향력을 조절하는 매개변수이고
#편향은 얼나나 쉽게 활성화(결과를 1로 출력)하느냐를 조절하느 매개 변수이다.
print(AND(0,0)) #0을 출력
print(AND(1,0)) #0을 출력
print(AND(0,1)) #0을 출력
print(AND(1,1)) #1을 출력
