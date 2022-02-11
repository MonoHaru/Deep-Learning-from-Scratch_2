# 1.3 신경망의 학습

# 1.3.1 손실 함수

# 1.3.2 미분과 기울기
'''
기울기 = 벡터의 각 원소에 대한 미분을 정리한 것
행렬과 그 기울기의 형상이 같다.
'''

# 1.3.3 연쇄 법칙
# 오차역전파법(back-propagation)
# 연쇄 법칙(chain rule): 합성함수에 대한 미분의 법칙
# 아무리 많은 함수를 연결하더라도, 그 미분은 개별 함수의 미분을 이용해 구할 수 있다.
# 각 함수의 국소적인 미분을 계산할 수 있다면 그 값들을 곱해서 전체의 미분을 구할 수 있다.

# 1.3.4 계산 그래프
# 계산 그래프: 계산 과정의 시각화

# 덧셈 노드
# 덧셈 노드는 상류로부터 받은 값에 1을 곱하여 하류로 기울기를 전파한다.
# 즉, 상류로부터의 기울기를 그대로 흘리기만 한다.

# 곱셈 노드
# 곱셈 노드의 역전파는 '상류로부터 받은 기울기'에 '순전파 시의 입력을 서로 바꾼 값'을 곱한다.
# 텐서는 다른 원소들과는 독립적으로, '원소별 연산'을 수행한다.

# 분기 노드(복제 노드)
# 분기 노드의 역전파는 상류에서 온 기울기들의 '합'이다.

# Repeat 노드
# 2개로 분기하는 분기 노드의 일반화
# N개의 분기
import numpy as np
D, N = 8, 7
x = np.random.randn(1, D)               # 입력
y = np.repeat(x, N, axis=0)             # 순전파
dy = np.random.randn(N, D)              # 무작위 기울기
dx = np.sum(dy, axis=0, keepdims=True)  # 역전파

# MatMul 노드
# 곱셈의 역전파는 '순전파 시의 입력을 서로 바꾼 값'을 사용한다.
# 행렬의 역전파도 '순전파 시의 입력을 서로 바꾼 행렬'을 사용한다.
# grads[0][...]: 생략기호
# a = b와 a[...] = b의 차이: 생략 기호는 데이터를 덮어쓰기 떄문에 변수가 가르키는 메모리 위치는 변하지 않는다.

# 1.3.5 기울기 도출과 역전파 구현
# common/layers.py/Sigmoid 계층
# common/layers.py/Affine 계층
# common/layers.py/Softmax with Loss
# Softmax 계층의 역전파는 자신의 출력과 정답 레이블의 차이이다.
# 신경망의 연전파는 이 차이(오차)를 앞 계층에 전해주는 것으로, 신경망 학습에서 아주 중요한 성질이다.

# 1.3.6 가중치 갱신
# common/optimizer.py/SGD: 확률적 경사하강법
from common.optimizer import SGD
from forward_net import TwoLayerNet
model = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000):
    '''
        x_batch, t_batch = get_mini_batch(...)  # 미니배치 획득
        loss = model.forward(x_batch, t_batch)
        model.backward()
        optimizer.update(model.loss, model.grads)
    '''