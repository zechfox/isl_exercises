import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, Ridge, Lasso
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

#Exercise 9
print('Exercise 9 ')
fileName = r'../dataSet/College.csv'
dfRaw = pd.read_csv(fileName).drop(['Unnamed: 0'], axis=1)
dfRaw['Private'] = pd.Categorical(dfRaw['Private']).codes
predictorList = list(dfRaw.columns)
predictorList.remove('Apps')

#Question A
print('(a) split train and test set as half-and-half.')
#make sure same random split on data set
np.random.seed(3)
trainSet, validationSet = train_test_split(dfRaw, train_size=0.5)
#Question B
print('(b) ')
model = LinearRegression()
model.fit(trainSet[predictorList], trainSet['Apps'])
print("Mean squared error: %.2f"%np.mean(model.predict(validationSet[predictorList]) - validationSet['Apps']) ** 2)

#Question C
print('(c) ')
def RidgeLambda(alpha, trainSet, validationSet):
  ridgeModel = Ridge(alpha)
  ridgeModel.fit(trainSet[predictorList], trainSet['Apps'])
  predictRidge = ridgeModel.predict(validationSet[predictorList])
  error_rate = np.mean((predictRidge - validationSet['Apps']) ** 2)
  return error_rate

ridgeTrainSet, ridgeValidationSet = train_test_split(trainSet, train_size=0.5)
alphaList = list(np.arange(0.01, 1, 0.001))
errorRateList = []
for alpha in alphaList:
  errorRate = RidgeLambda(alpha, ridgeTrainSet, ridgeValidationSet)
  errorRateList.append(errorRate)
testError = RidgeLambda(alphaList[errorRateList.index(min(errorRateList))], trainSet, validationSet)
print('Test error is: %.2f'%testError)
#Question D
print('(d) ')
def LassoLambda(alpha, trainSet, validationSet):
  lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
  lassoreg.fit(trainSet[predictorList], trainSet['Apps'])
  predict_lasso = lassoreg.predict(validationSet[predictorList])
  error_rate = np.mean((predict_lasso - validationSet['Apps']) ** 2)
  return error_rate
lassoTrainSet, lassoValidationSet = train_test_split(trainSet, train_size=0.5)
alphaList = list(np.arange(0.01, 1, 0.001))
errorRateList = []
for alpha in alphaList:
  errorRate = LassoLambda(alpha, lassoTrainSet, lassoValidationSet)
  errorRateList.append(errorRate)
testError = LassoLambda(alphaList[errorRateList.index(min(errorRateList))], trainSet, validationSet)
print('Test error is: %.2f'%testError)

#Question E
print('(e) ')
n_components = list(np.arange(1, len(predictorList)))
pcaTrainSet, pcaValidationSet = train_test_split(trainSet, train_size=0.5)
lr = LinearRegression()
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('lr', lr)])  
estimator = GridSearchCV(pipe, dict(pca__n_components=n_components))
estimator.fit(pcaTrainSet[predictorList], pcaTrainSet['Apps'])
predictPCA = estimator.predict(validationSet[predictorList])
testError = np.mean((predictPCA - validationSet['Apps']) ** 2)
print('Test error is: %.2f'%testError)

#Question F
print('(f) ')
def PLSCrossValidation(n_components, trainSet, validationSet):
  pls = PLSRegression(n_components=n_components)
  pls.fit(trainSet[predictorList], trainSet['Apps'])
  predictPls = pls.predict(validationSet[predictorList])
  different = predictPls.flat - validationSet['Apps']
  error_rate = np.mean(different ** 2)
  return error_rate
plsTrainSet, plsValidationSet = train_test_split(trainSet, train_size=0.5)
n_componentsList = list(np.arange(1, len(predictorList)))
errorRateList = []
for n_components in n_componentsList:
  errorRate = PLSCrossValidation(n_components, plsTrainSet, plsValidationSet)
  errorRateList.append(errorRate)
testError = PLSCrossValidation(n_componentsList[errorRateList.index(min(errorRateList))], trainSet, validationSet)
print('Test error is: %.2f'%testError)
#Question G
print('(g) It seems normal linear model fit the result very well. ')

