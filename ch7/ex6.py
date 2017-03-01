import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import cross_validation
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#Exercise 6
print('Exercise 6 ')
fileName = r'../dataSet/Wage.csv'
df = pd.read_csv(fileName)
X = df['age']
Y = df['wage']


#Question A
print('(a) ')
degrees = np.arange(1, 11)
scores = [np.mean(cross_validation.cross_val_score(Pipeline([('ply', PolynomialFeatures(degree=degree)),
                                                     ('linear', LinearRegression())]), 
                                           X[:, np.newaxis], Y, cv=10, scoring='r2'))
          for degree in degrees]
plt.plot(degrees, scores)
print('from the figure1, it can be tell degree=4 is the best fitting.')
#Quesiton B
print('(b) ignore.')
#0. sort df by 'age'
#1. divide x range into M pieces
#2. fit each piece with LR
#3. run cross validation on each piece
#4. mean this M pieces to get score
#5. get out the model of best score
dfSorted = df.sort_values('age')
maxAge = max(dfSorted['age'])
minAge = min(dfSorted['age'])
#the r-square is negtive because the model fit is bad
bestScore = -1
bestM = 0 
bestPredict = []
sum = 0
for m in np.arange(2, 10):
  interval = math.ceil((maxAge - minAge) / m)
  modelList = []
  scoresList = []
  predictList = []
  sum = 0
  for i in np.arange(m):
    piecesDf = dfSorted[(dfSorted['age'] >= (minAge + i * interval)) & (df['age'] < (minAge + (i+1) * interval))]
    if piecesDf.empty:
      continue
    X = piecesDf['age']
    Y = piecesDf['wage']
    predict = cross_validation.cross_val_predict(LinearRegression(), 
                                             X[:, np.newaxis], Y, cv=10)
    scores = metrics.r2_score(Y, predict)
    scoresList.append(scores)
    predictList.append(predict)
   
  if np.mean(scoresList) > bestScore:
    bestScore = np.mean(scoresList)
    bestPredict = predictList 
    bestM = m

print('The model fitted by step function is bad, it\'s not a good way to fit this set of data.')
plt.figure(2)
plt.plot(dfSorted['age'], np.hstack(bestPredict), '-')
plt.plot(dfSorted['age'], dfSorted['wage'], 'o')
plt.show()

