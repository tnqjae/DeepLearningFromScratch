#나쁜 구현 예
def numberical_diff(f, x):
    h = 1e-50 #너무 작아서 컴퓨터에서 생략을 해버린다. (실제 값은 0.00000..1이지만, 출력이 0.0이 나옴)
    return (f(x+h) - f(x)) / h # 수식을 그대로 표현

#위 코드를 계선한 코드
def improve_numberical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)