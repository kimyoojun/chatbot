import nltk
nltk.download()
text = nltk.word_tokenize("Is it possible distinguishing cats and dogs")
text



nltk.download('averaged_perceptron_tagger')


nltk.pos_tag(text)



nltk.download('punkt')
string1 = "my favorite subject is math"
string2 = "my favorite subject is math, english, economic and computer science"
nltk.word_todenize(string1)


nltk.word_tokenize(string2)



from konlpy.tag import Komoran
komoran = Komoran()
print(komoran.morphs('딥러닝이 쉽나요? 어렵나요?'))



print(komoran.pos('소파 위에 있는 것이 고양이인가요? 강아지인가요?'))



import pandas as pd
df = pd.read_csv('..\chap09\data\class2.csv')
df


df.isnull().sum()



df.isnull().sum() / len(df)



df = df.dropna(how='all')
print(df)



df1 = df.dropna()
print(df1)



df2 = df.fillna(0)
print(df2)



df['x'].fillna(df['x'].mean(), inplace=True)
print(df)






