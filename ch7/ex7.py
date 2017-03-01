import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import cross_validation
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#Exercise 7
print('Exercise 7 ')
fileName = r'../dataSet/Wage.csv'
df = pd.read_csv(fileName)
df['maritl'] = pd.Categorical(df['maritl']).codes
X = df['maritl']
Y = df['wage']


#Question A
print('polynomial ')
degrees = np.arange(1, 11)
scores = [np.mean(cross_validation.cross_val_score(Pipeline([('ply', PolynomialFeatures(degree=degree)),
                                                     ('linear', LinearRegression())]), 
                                           X[:, np.newaxis], Y, cv=10, scoring='r2'))
          for degree in degrees]
plt.plot(degrees, scores)
print('from the figure1, it can be tell degree=4 is the best fitting.')
#Quesiton B
print('step wise')
#0. sort df by 'age'
#1. divide x range into M pieces
#2. fit each piece with LR
#3. run cross validation on each piece
#4. mean this M pieces to get score
#5. get out the model of best score
dfSorted = df.sort_values('maritl')
maxAge = max(dfSorted['maritl'])
minAge = min(dfSorted['maritl'])
#the r-square is negtive because the model fit is bad
bestScore = -1
bestM = 0 
bestPredict = []
sum = 0
for m in np.arange(2, 10):
  interval = math.ceil((maxAge - minAge) / m) + 0.1
  modelList = []
  scoresList = []
  predictList = []
  sum = 0
  for i in np.arange(m):
    piecesDf = dfSorted[(dfSorted['maritl'] >= (minAge + i * interval)) & (df['maritl'] < (minAge + (i+1) * interval))]
    if piecesDf.empty:
      continue
    X = piecesDf['maritl']
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

print('It can be told that married people earn more money than non-married people.')
plt.figure(2)
plt.plot(dfSorted['maritl'], np.hstack(bestPredict), '-')
plt.plot(dfSorted['maritl'], dfSorted['wage'], 'o')
plt.show()

