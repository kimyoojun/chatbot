import numpy as np
%matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import _pca
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# 아래 상대 경로에 오류가 발생한다면 절대 경로로 지정해 주세요.
glove_file = datapath('C:\chatbot\python\pythorch\data2\glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("C:\chatbot\python\pythorch\data2\glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)



model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
model.most_similar('bill')



model.most_similar('cherry')



model.most_similar(negative=['cherry'])



result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))



def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]
analogy('australia', 'beer', 'france')



analogy('tall', 'tallest', 'long')



print(model.doesnt_match("breakfast cereal dinner lunch".split()))