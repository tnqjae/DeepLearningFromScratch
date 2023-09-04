class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
#mask는 True/False로 수성된 넘파이 배열
#순전파에서는 x의 원소 값이 0 이하인 인덱스는 True, 그 외는 False 유지
