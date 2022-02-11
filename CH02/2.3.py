# 2.3 통계 기반 기법
# 통계 기반 기법에서는 말뭉치(corpus)를 사용한다
# 말뭉치란 조건하에 수집괸 텍스트 데이터를 의미한다
# 통계 기반 기법의 목표는 말뭉치에서 자동으로, 그리고 효율적으로 그 핵심을 추출하는 것이다

# 2.3.1 파이썬으로 말뭉치 처리하기
# 전처리(preprocessing): 텍스트 데이터를 단어로 분할하고 그 분할된 단어들을 단어 ID 목록으로 변환하는 것
text = 'You say goodbye and I say hello.'
text = text.lower()                 # lower(): 모두 소문자로 변환함
text = text.replace('.', ' .')
text
words = text.split(' ')             # split(' '): ' '을 기준으로 분할함
words

# 텍스트를 그대로 조작하기엔 불편함
# 따라서 단어에 ID를 부여하고, ID의 리스트로 이용할 수 있도록 손질함
# 파이썬 딕셔너리를 이용하여 단어 ID와 단어를 짝지어주는 대응표를 작성함
word_to_id = {}
id_to_word ={}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print(id_to_word)
print(word_to_id)
print(id_to_word[1])
print(word_to_id['hello'])

# '단어 목록'을 '단어 ID 목록'으로 변경, 그리고 넘파이 배열로 변환
import numpy as np
corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
print(corpus)

from common.util import preprocess
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

# 2.3.2 단어의 분산 표현
# 분산 표현(ditributional representation): '단어의 의미'를 정확하게 파악할 수 있는 벡터 표현

# 2.3.3 분포 가설
# 자연어 처리의 중요 기법 '분포 가설(ditributional hypothesis): '단어의 의미는 주변 단어에 의해 형성된다.'
# 즉, 단어 자체에는 의미가 없고, 그 단어가 사용된 맥락(context)이 의미를 형성한다.
# '맥락'이란 특정 단어를 중심에 둔 준변 단어
# 윈도우 크기(window size): 맥락의 크기(주변 단어를 몇 개나 포함할지)

# 2.3.4 동시발생 행렬
# 통계 기반(statistical based): 어떤 단어에 주목했을 때, 그 주변에 어떤 단어가 몇 번 등장하는지를 세어 집계하는 방법
# import sys
# sys.path.append('..')
import numpy as np
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)

# 표의 각 행은 해당 단어를 표현한 벡터이다.
# 이 표가 행렬의 띤다는 뜻에서 '동시발생 행렬(co-occurrence matrix)'라고 부름
C = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0]
], dtype=np.int32)
print(C[0])                             # ID가 0인 단어의 벡터 표현
print(C[4])                             # ID가 4인 단어의 벡터 표현
print(C[word_to_id['goodbye']])         # 'goodbye'의 벡터 표현

# 2.3.5 벡터 간 유사도
# 코사인 유사도(cosine similarity): 단어 벡터의 유사도를 나타낼 때 사용함
# 코사인 유사도는 '두 벡터가 가르키는 방향이 얼마나 비슷한가'를 측정한다
# 두 벡터의 방향이 완전히 같다면 코사인 유사도는 1, 완전히 반대라면 -1이 된다

# "you"와 "i(=I)"의 유사도를 구하는 코드
# import sys
# sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]       # "you"의 단어 벡터
c1 = C[word_to_id['i']]         # "i"의 단어 벡터
print(cos_similarity(c0, c1))

# 2.3.6 유사 단어의 랭킹 표시
# 검색어와 비슷한 단어를 유사도 순으로 출력하는 함수
# common.util.most_similar()
# "you"를 검색어로 지정해 유사한 단어 출력하는 코드
# import sys
# sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similar

text = 'You say goodbte and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C ,top=5)