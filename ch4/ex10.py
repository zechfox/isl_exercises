import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy import stats
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier

fileName = r'../dataSet/Weekly.csv'
df = pd.read_csv(fileName)
dfNew = df.loc[:, 'Year':'Direction']
#categorical the Direction column
dfNew['Direction'] = pd.Categorical(dfNew['Direction']).codes
print('Exercise 10 Anser:')
#Question A
print('(a) see figure1')
#describe() is same as summary() of R
print(dfNew.describe())
pd.scatter_matrix(dfNew, alpha=0.5)

#Question B
print('(b) Lag2 seems have some statistical significance with P>|z|=%3.')
model = smf.logit(formula='Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume', data=dfNew)
res = model.fit()
print(res.summary())

#Question C
#pred_table, row is actual, column is predict
#we can also use crosstab of pandas to generate confusion matrix
print(res.pred_table())
print('(c) correct rate of prediction: 56.1%, mainly because much wrong prediction in UP.')

#Question D
#TODO:why following get error?
#trainingSet = dfNew.loc[lambda dfNew: (dfNew.Year < 2008) and (dfNew.Year > 1990), :]
print('(d) ')
trainingSet = dfNew.loc[lambda df: df.Year < 2009, :]
testSet = dfNew.loc[lambda df: df.Year >= 2009, :]
model_D = smf.logit(formula='Direction~Lag2', data=trainingSet)
result_D = model_D.fit()
predict_testSet = result_D.predict(testSet)
predict_testSet = np.array([(lambda x:int(1) if (x > 0.5) else int(0))(x) for x in predict_testSet])
actual_testSet = testSet['Direction'].values
res = pd.crosstab(predict_testSet, actual_testSet)
print(res)
correct_rate= np.mean(actual_testSet == predict_testSet)
print('overall fraction of correct predictions: {correct_rate}'.format(correct_rate=correct_rate))

#prepare data for following questions
X = trainingSet['Lag2'].values
y = trainingSet['Direction'].values
training_zeros = np.zeros(X.shape)
test_zeros = np.zeros(testSet['Lag2'].values.shape)
testSet = testSet['Lag2'].values
testSet = np.column_stack((testSet, test_zeros))

#Question E
print('(e) ')
#the LDA requires 2 features at least, I didn't find a way to set LDA use 1 feature
#I have to append 0 to training data to increase the dimension
#np.concatenate, np.hstack and np.insert didn't work, because axis has to larger than 1.
#if use np.concatenate, np.hstack and np.insert, np.expand_dims should be used firstly.
X = np.column_stack((X, training_zeros))
lda = LDA(n_components=1)
lda = lda.fit(X, y)
predict_LDA= lda.predict(testSet)
res = pd.crosstab(predict_LDA, actual_testSet)
print(res)
correct_rate= np.mean(actual_testSet == predict_LDA)
print('overall fraction of correct predictions: {correct_rate}'.format(correct_rate=correct_rate))

#Question F
print('(f) ')
qda = QDA()
qda = qda.fit(X, y)
predict_QDA = qda.predict(testSet)
res = pd.crosstab(predict_QDA, actual_testSet)
print(res)
correct_rate= np.mean(actual_testSet == predict_QDA)
print('overall fraction of correct predictions: {correct_rate}'.format(correct_rate=correct_rate))

#Question G
print('(g) ')
knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(X, y)
predict_KNN = knn.predict(testSet)
res = pd.crosstab(predict_KNN, actual_testSet)
print(res)
correct_rate= np.mean(actual_testSet == predict_KNN)
print('overall fraction of correct predictions: {correct_rate}'.format(correct_rate=correct_rate))


#Question H
print('(h) LDA and ligistic regression has similar error rates.')

#Question I
print('(i) ignore.')
plt.show()
