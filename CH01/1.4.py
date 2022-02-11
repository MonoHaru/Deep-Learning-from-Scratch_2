# 1.4 신경망으로 문제를 풀다

# 1.4.1 스파이럴 데이터셋
# import sys
# sys.path.append('..')
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape)     # (300, 2)
print('t', t.shape)     # (300, 3)

# 1.4.2 신경망 구현
# import sys
# sys.path.append('..')
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from CH01.two_layer_net import TwoLayerNet

# 1.4.3 학습용 코드
import CH01.train_custom_loop

# 1.4.4 Trainer 클래스
from common.trainer import Trainer
'''
    model = TwoLayerNet(...)
    optimizer = SGD(lr=1.0)
    trainer = Trainer(model, optimizer)
    - fit() 메서드를 호출해 학습을 시작함
    - plot() 메서드를 제공함
'''
import CH01.train