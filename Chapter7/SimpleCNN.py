import numpy as np
from util import im2col
from collections import OrderedDict
from Conv import Convolution
class SimpleConvNet:
    #초기화
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30 ,'filter_size':5, 'pad':0,'stride':1},
                 hidden_size = 100, output_size=10, weight_init_std=0.01):
            filter_num = conv_param['filter_num']
            filter_size = conv_param['filter_size']
            filter_pad = conv_param['pad']
            filter_stride = conv_param['stride']
            input_size = input_dim[1]
            conv_output_size = (input_size - filter_size + 2 * filter_pad) / \
                                filter_stride + 1
            pool_output_size = int(filter_num * (conv_output_size / 2) *
                                   (conv_output_size / 2))
    
    #합성곱 계층의 출력 크기를 계산
            self.params ={}

            #1층
            self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_stride)
            self.params['b1'] = np.zeros(filter_num)

            #2층
            self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
            self.params['b2'] = np.zeros(hidden_size)
            
            #3층
            self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
            self.params['b3'] = np.zeros(output_size)

            self.layers = OrderedDict()
            self.layers = Convolution(self.params['W1'],
                                      self.params['b1'],
                                      conv_param['stride'],
                                      conv_param['pad'])
