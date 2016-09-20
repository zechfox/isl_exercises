import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

fileName = r'../dataSet/Boston.csv'
df = pd.read_csv(fileName).drop(['Unnamed: 0'], axis=1)
print(df.head(5))

#Exercise 9
print('Exercise 9 ')

#Question A
miu = np.mean(df['medv'])
print('(a) mean of medv is {miu}'.format(miu=miu))

#Question B
stdErr = np.std(df['medv']) / np.sqrt(df['medv'].shape[0])
print('(b) standard error of miu is {stdErr}'.format(stdErr=stdErr))

#Question C
def calculateMean(data, indexArray):
  return np.mean(data.iloc[indexArray])

def bootStrap(data, function, R):
  maxIndex = data.shape[0]
  result = pd.Series(np.zeros(maxIndex))
  for loop in range(R):
    randomIndex = np.random.randint(maxIndex, size=maxIndex)
    mean = function(data, randomIndex)
    result[loop] = mean
  return np.std(result)
stdErrBootStrap = bootStrap(df['medv'], calculateMean, 1000)
print('(c) standard error of miu using bootstrap is {stdErrBootStrap}'.format(stdErrBootStrap=stdErrBootStrap))

#Question D
bottom = miu - 2*stdErr
top = miu + 2*stdErr
print('(d) 95% confidence interval is [{bottom}, {top}]'.format(bottom=bottom, top=top))

#Question E
median = np.median(df['medv'])
print('(e) meadian is {median}'.format(median=median))

#Question F
def calculateMedian(data, indexArray):
  return np.median(data.iloc[indexArray])
stdErrBootStrap = bootStrap(df['medv'], calculateMedian, 1000)
print('(f) standard error of miu_median is {stdErrBootStrap}'.format(stdErrBootStrap=stdErrBootStrap))

#Question G
tenthPercentile = np.percentile(df['medv'], q=10)
print('(g) tenth percentile of medv is {tenthPercentile}'.format(tenthPercentile=tenthPercentile))

#Question H
def calculate10Percentile(data, indexArray):
  return np.percentile(data.iloc[indexArray], q=10)

tenthPercentileBootStrap = bootStrap(df['medv'], calculate10Percentile, 1000)
print('(h) standard error of tenth percentile using bootstrap is {tenthPercentileBootStrap}'.format(tenthPercentileBootStrap=tenthPercentileBootStrap))
