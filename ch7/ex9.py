import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#Exercise 9
print('Exercise 9 ')
fileName = r'../dataSet/Boston.csv'
df = pd.read_csv(fileName)
X = df['dis']
Y = df['nox']
print(X.head(5))
print(Y.head(5))
#Question A
print('(a) ')
degrees = np.arange(1, 6)
scores = [np.mean(cross_validation.cross_val_score(Pipeline([('ply', PolynomialFeatures(degree=degree)),
                                                             ('linear', LinearRegression())]),
                                                   X[:, np.newaxis], Y, cv=10, scoring='r2'))
         for degree in degrees]
plt.plot(degrees, scores) 
print('It can be told that degree = 3 has best performance.')


#Question B
print('(b) ')
degrees = np.arange(1, 11)
scores = [Pipeline([('ply', PolynomialFeatures(degree=degree)),
                    ('linear', LinearRegression())]).fit(X[:, np.newaxis], Y).score(X[:, np.newaxis], Y)
         for degree in degrees]
print('It can be told from the score that the RSS increased as degree.')

#Question C
print('(c) same as (a). ')

#Question D
print('(d) ')


plt.show()
