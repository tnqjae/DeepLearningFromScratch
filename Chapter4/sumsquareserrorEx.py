import numpy as np
from SumSquaresError import sum_squares_error
#정답은 2번 인덱스
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

#예 1: '2'일 확률이 가장 높다고 추정함(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
p1 = sum_squares_error(np.array(y), np.array(t))
print(p1)# 출력값 : 0.09750000000000003

#예2 : 7번째 인덱스일 확률이 가장 높다고 추정(0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
p2 = sum_squares_error(np.array(y), np.array(t))
print(p2) # 출력값 : 0.5975
