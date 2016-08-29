import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier

fileName = r'../dataSet/Auto.csv'
df_raw = pd.read_csv(fileName)
df_raw_numeric = df_raw.apply(pd.to_numeric, args=('coerce',))
df = df_raw_numeric.loc[:, 'mpg':'origin'].dropna(0, 'any')

mpg_median = df['mpg'].median()

mpg01 = df['mpg'].apply(lambda x: 1 if x > mpg_median else 0)
df = df.assign(mpg01=mpg01)
print('Exercise 11: ')
#Question A
print('(a) ')
print(df.describe())


#Question B
print('(b) From the scatter plot, it\'s hard to tell the relationship between mpg01 and other features, becasue mpg01 is 0 or 1.\
we can tell the relationship from correlation matrix. So cylinders, displacement, horsepower and weight are useful.')
pd.scatter_matrix(df, alpha=0.5)
print(df.corr())


#Question C
print('(c) split first 80% as training set, others as test set.')
print(df.shape[0])
trainingSet = df.head(round(df.shape[0] * 0.8))
testSet = df.tail(round(df.shape[0] * 0.2))
trainingData = trainingSet.loc[:, 'cylinders':'weight']
trainingResponse = trainingSet['mpg01']
testData = testSet.loc[:, 'cylinders':'weight']
testResponse = testSet['mpg01']


#Question D
print('(d) ')
lda = LDA()
lda = lda.fit(trainingData, trainingResponse)
predict_LDA = lda.predict(testData)
res = pd.crosstab(predict_LDA, testResponse.values)
print(res)
error_rate = np.mean(predict_LDA != testResponse.values)
print('LDA test error is {error_rate}'.format(error_rate=error_rate))

#Question E
print('(e) ')
qda = QDA()
qda = qda.fit(trainingData, trainingResponse)
predict_QDA = qda.predict(testData)
res = pd.crosstab(predict_QDA, testResponse.values)
print(res)
error_rate = np.mean(predict_QDA != testResponse.values)
print('QDA test error is {error_rate}'.format(error_rate=error_rate))

#Question F
print('(f) ')
lr = LogisticRegression()
lr = lr.fit(trainingData, trainingResponse)
predict_LR = lr.predict(testData)
res = pd.crosstab(predict_LR, testResponse.values)
print(res)
error_rate = np.mean(predict_LR != testResponse.values)
print('Logistic Regression test error is {error_rate}'.format(error_rate=error_rate))

#Question G
print('(g) K=2 seems perform best.')
knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(trainingData, trainingResponse)
predict_KNN = knn.predict(testData)
res = pd.crosstab(predict_KNN, testResponse.values)
print(res)
error_rate = np.mean(predict_KNN != testResponse.values)
print('KNN k=1, test error is {error_rate}'.format(error_rate=error_rate))
knn = KNeighborsClassifier(n_neighbors=2)
knn = knn.fit(trainingData, trainingResponse)
predict_KNN = knn.predict(testData)
res = pd.crosstab(predict_KNN, testResponse.values)
print(res)
error_rate = np.mean(predict_KNN != testResponse.values)
print('KNN k=2, test error is {error_rate}'.format(error_rate=error_rate))
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(trainingData, trainingResponse)
predict_KNN = knn.predict(testData)
res = pd.crosstab(predict_KNN, testResponse.values)
print(res)
error_rate = np.mean(predict_KNN != testResponse.values)
print('KNN k=3, test error is {error_rate}'.format(error_rate=error_rate))

plt.show()
