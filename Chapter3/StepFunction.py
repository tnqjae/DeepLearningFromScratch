#기본적인 계단 함수 구현
import numpy as np

def step_function(x):
    y = x > 0 #해당 코드를 실행하게 되면 np.array 값을 불러와 참 거짓 판단을하고, 그 값을 y에 저장
    return y.astype(np.int)#이 코드를 실행시키면 y의 값이 [true, false, false]인 값을 [1, 0, 0]으로 바꿔준다.

#계단함수 그리기
import matplotlib.pylab as plt

def Short_step_funcion(x):
    return np.array(x > 0, dtype=np.int)#위 코드를 1줄로

x = np.arange(-5.0, 5.0, 0.1)
y = Short_step_funcion(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1)#y축의 범위를 지정한다.
plt.show()
