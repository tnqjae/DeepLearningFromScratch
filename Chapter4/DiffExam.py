import numpy as np
import matplotlib.pylab as plt
from Differential import improve_numberical_diff

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def tangent_line(f, x):
    d = improve_numberical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1) #0부터 20미만 까지 0.1씩 채움
y = function_1(x)

#계산한 수치값
print(improve_numberical_diff(function_1, 5)) # print : 0.1999999999990898
print(improve_numberical_diff(function_1, 10)) # print : 0.2999999999986347

#function_1에 대한 그래프를 그리고
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y)

#수치 미분 값으로 접선을 그려 봄
tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show() # 그래프를 확인해보면 수치 미분 값으로 그린 직선은 5에서 접선을 이룸