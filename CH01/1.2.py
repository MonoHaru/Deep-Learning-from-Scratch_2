# 1.2 신경망 추론

# 1.2.1 신경망 추론 전체 그림
import numpy as np
W1 = np.random.randn(2, 4)   # 가중치
b1 = np.random.randn(4)      # 편향
x = np.random.randn(10, 2)   # 입력
h = np.matmul(x, W1) + b1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = sigmoid(x)

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(10, 2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.matmul(x, W1) + b1
a = sigmoid(h)
s = np.matmul(a, W2) + b2

# 1.2.2 계층으로 클래스화 및 순전파 구현
from CH01.forward_net import *

a = ['A', 'B']
a += ['C', 'D']
print(a)

x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)