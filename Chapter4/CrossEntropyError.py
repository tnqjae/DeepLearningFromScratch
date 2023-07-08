import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))#delta를 더해주는 이유는 ln에 값에 0이 들어가는 것을 방지


