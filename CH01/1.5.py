# 1.5 계산 고속화
# 신경망 고속화

# 1.5.1 비트 정밀도
import numpy as np
a = np.random.randn(3)
print(a.dtype)      # float64, 64비트 부동소수점

# 이 책에선 일반적으로 32비트 부동소수점 수를 우선으로 사용함
b = np.random.randn(3).astype(np.float32)
print(b.dtype)

c = np.random.randn(3).astype('f')
print(c.dtype)

'''
    가중치를 파링레 저장할 때는 16비트 부동소수점 수를 사용함
    용량의 절반만을 사용하기 때문에
'''

# 1.5.2 GPU(쿠파이)
# 쿠파이는 GPU를 이용해 병렬 계산을 수행하는 라이브러리, 엔비디아의 GPU에서만 동작함
import cupy as cp
x = cp.arange(6).reshape(2, 3).astype('f')
print(x)
print(x.sum(axis=1))