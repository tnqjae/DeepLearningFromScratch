import numpy as np

#곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):#순전파
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):#역전파
        dx = dout * self.y # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy
    
#덧셈 계층
class AddLayer:
    def __init__(self):
        pass#아무 기능도 하지 않음
    
    def forward(self, x, y):#순전파
        out = x + y

        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy