import numpy as np

def numberical_gradient(f, x):
    h = 1e-4 #0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        #f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #값 복원

    return grad
 