from nltk import sent_tokenize
text_sample = 'By promptly disclosing medical errors and offering earnest apologies and fair compensation, doctors hope to make it easier to learn from mistakes and relieve the patient's anger.'.
tokenized_sentences = sent_tokenize(text_sample)
print(tokenized_sentences)



from nltk import word_tokenize
sentence = "This book is for deep learning learners"
words = word_tokenize(sentence)
print(words)



from nltk.tokenize import WordPunctTokenizer
sentence = "it's nothing that you don't already know except most people aren't aware of how their inner world works."
words = WordPunctTokenizer().tokenize(sentence)
print(words)



import csv
from konlpy.tag import Okt
from gensim.models import word2vec

f = open(r'..\data\ratings_train.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
rdw = list(rdr)
f.close()



twitter = Okt()

result = []
for line in rdw:
    malist = twitter.pos(line[1], norm=True, stem=True)
    r = []
    for word in malist:
        if not word[1] in ["Josa","Eomi","Punctuation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    result.append(rl)
    print(rl)



with open("NaverMovie.nlp", 'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))



mData = word2vec.LineSentence("NaverMovie.nlp")
mModel = word2vec.Word2Vec(mData, vector_size=200, window=10, hs=1, min_count=2, sg=1)
mModel.save("NaverMovie.model")



import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

sample_text = "One of the first things that we ask ourselves is what are the pros and cons of any task we perform."
text_tokens = word_tokenize(sample_text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words(
                    'english')]
print("불용어 제거 미적용:", text_tokens, '\n')
print("불용어 제거 적용:", tokens_without_sw)



from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

print(stemmer.stem('obesses'), stemmer.stem('obssesed'))
print(stemmer.stem('standardizes'), stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))
print(stemmer.stem('absentness'), stemmer.stem('absently'))
print(stemmer.stem('tribalical'), stemmer.stem('tribalicalized'))



from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('obsesses'), stemmer.stem('obsessed'))
print(stemmer.stem('standardizes'), stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))
print(stemmer.stem('absentness'), stemmer.stem('absently'))
print(stemmer.stem('tribalical'), stemmer.stem('tribalicalized'))



import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

print(stemmer.stem('obsesses'), stemmer.stem('obsessed'))
print(lemma.lemmatize('standardizes'), lemma.lemmatize('standardization'))
print(lemma.lemmatize('national'), lemma.lemmatize('nation'))
print(lemma.lemmatize('absentness'), lemma.lemmatize('absently'))
print(lemma.lemmatize('tribalical'), lemma.lemmatize('tribalicalized'))



print(lemma.lemmatize('obsesses', 'v'), lemma.lemmatize('obsessed', 'a'))
print(lemma.lemmatize('standardizes', 'v'), lemma.lemmatize('standardization', 'n'))
print(lemma.lemmatize('national', 'a'), lemma.lemmatize('nation', 'n'))
print(lemma.lemmatize('absentness', 'n'), lemma.lemmatize('absently', 'r'))
print(lemma.lemmatize('tribalical', 'a'), lemma.lemmatize('tribalicalized', 'v'))



