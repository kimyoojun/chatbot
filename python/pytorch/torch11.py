import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('pythorch/data/train.csv',index_col='PassengerId')
print(df.head())

df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna()
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))    

print(pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Surcical', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
))


