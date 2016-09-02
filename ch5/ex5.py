import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

fileName = r'../dataSet/Default.csv'
df = pd.read_csv(fileName).drop(['Unnamed: 0'], axis=1)
df['default'] = pd.Categorical(df['default']).codes
df['student'] = pd.Categorical(df['student']).codes
print('Exercise 5 ')
#Question A
lr = LogisticRegression()
lr = lr.fit(df.loc[:, 'balance':'income'], df.loc[:, 'default'])
print('(a) Fitted')

#Question B
def questionB():
  #i. validation set approach randomly split the data set half vs half
  trainSet, validationSet = train_test_split(df, train_size=0.5)
  #ii. fit LR only with train set
  lr = LogisticRegression()
  lr = lr.fit(trainSet[['balance','income']], trainSet['default'])
  #iii. predict with validation set
  predict_LR = lr.predict(validationSet.loc[:, 'balance':'income'].values)
  #iv. compute validation error rate
  error_rate = np.mean(predict_LR != validationSet.loc[:, 'default'])
  return error_rate
error_rate = questionB()
print('(b) error rate is {error_rate}%'.format(error_rate=error_rate*100))

#Question C
error_rate1 = questionB()
error_rate2 = questionB()
error_rate3 = questionB()

print('(c) 3 times result: {er1}, {er2}, {er3}. It seems the error rate around 3.5%.'.format(er1=error_rate1, er2=error_rate2, er3=error_rate3))

#Question D
trainSet, validationSet = train_test_split(df, train_size=0.5)
lr = LogisticRegression()
lr = lr.fit(trainSet.loc[:, 'student':'income'], trainSet.loc[:, 'default'])
predict_LR = lr.predict(validationSet.loc[:, 'student':'income'].values)
error_rate = np.mean(predict_LR != validationSet.loc[:, 'default'])
print('(d) error rate is {error_rate}%, not reduce the test error.'.format(error_rate=error_rate*100))

