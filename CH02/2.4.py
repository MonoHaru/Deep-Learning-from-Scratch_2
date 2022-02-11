# 2.4 통계 기반 기법 개선하기

# 2.4.1 상호정보량
# '발생(고빈도 단어)' 횟수라는 특징은 좋은 특징은 아님
'''
    점별 상호정보량(PMI, Pointwise Mutual Information)
    PMI(x, y) = log2(P(x, y) / (P(x)P(y))
    P(x): x가 일어날 확률
    P(y): y가 일어날 확률
    P(x, y): x와 y가 동시에 일어날 확률
    문제점: 두 단어의 동시발생 횟수가 0이면 (-)무한대가 된다
'''
'''
    점별 상호정보량의 문제를 해결하기 위해 양의 상호정보량(PPMI, Positive PMI)를 사용함
    PPMI(x, y) = max(0, PMI(x, y))
    PMI가 음수일 경우 0으로 취급함
'''

# 동시발생 행렬을 PPMI 행렬로 변환하는 함수
# common.util.ppmi()
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)    # 유효 자릿수를 세 자리로 표시
print('동시 발생 행렬')
print(C)
print('-' * 50)
print('PPMI')
print(W)

'''
    이 헹렬의 원소 대부분은 0이다.
    즉, 벡터의 원소 대부분 '중요도'가 낮다는 뜻이다.
    이런 벡터는 노이즈에 약하고 견고하지 못하다는 단점이 있다.
    이 문제를 대체하고자 수행되는 기법이 '벡터의 차원 감소' 이다.
'''

# 2.4.2 차원 감소
# 차원 감소(dimensionality redution)는 '중요한 정보'는 최대한 유지하면서 벡터의 차원을 줄이는 방법이다.
'''
    특잇값분해(SVD, Singular Value Decomposition)
    SVD는 임의의 행렬을 세 행렬의 곱으로 분해한다.
    X = U * S * V.T
    U, V: 직교행렬(orthogonal matrix)이고, 그 열벡터를 서로 직교한다.
    S: 대각행렬(diagonal matrix, 대각성분 외에는 모두 0인 행렬)이다.
    U 행렬은 '단어공간', S 행렬은 '특잇값'이 큰 순서로 나열한다.
'''

# 2.4.3 SVD에 의한 차원 감소
# import sys
# sys.path.appen('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size,window_size=1)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print(C[0])     # 동시발생 행렬
print(W[0])     # PPMI 행렬
print(U[0])     # SVD
print(U[0, :2])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()

# 2.4.4 PTB 데이터셋
# 펜 트리뱅크(PTB, Penn Treebank): 주어진 기법의 품질을 측정하는 벤치마크로 자주 이용된다.
# 펜 트리뱅크를 사용하는 코드
# import sys
# sys.path.append('..')
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')

print('말뭉치 크기:', len(corpus))
print('corpus[:30]:', corpus[:30])
print()
print('id_to_word[0]:', id_to_word[0])
print('id_to_word[1]:', id_to_word[1])
print('id_to_word[2]:', id_to_word[2])
print()
print("word_to_id['car']:", word_to_id['car'])
print("word_to_id['happy']:", word_to_id['happy'])
print("word_to_id['lexus']:", word_to_id['lexus'])

# 2.4.5 PTB 데이터셋 평가
# PTB 데이터셋 평가 코드
# import sys
# sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('동시 발생 수 계산 ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('PPMI 계산 ...')
W = ppmi(C, verbose=True)

print('SVD 계산 ...')
try:
    # truncated SVD (빠르다!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    # SVD (느리다)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)