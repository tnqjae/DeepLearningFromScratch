class ReLU:
    def __init__(self):
        self.mask = None

    #순전파
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return 0
    
    #역전파
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx