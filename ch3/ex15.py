import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


fileName = '../dataSet/Boston.csv'
df = pd.read_csv(fileName)
#remove index column, inplace make the original data can be modified without creating a copy
df.drop(df.columns[0], axis = 1, inplace = True)

print('Exercise 15, Answers:')
predictor = df.drop(df.columns[0], axis =1)
print(predictor)
response = df['crim']
univariate_coef = np.empty([len(predictor.columns)])
print(univariate_coef)
#Question A
idx = 0
for column in predictor:
  model = sm.OLS(response, sm.add_constant(predictor[column]))
  result = model.fit()
  univariate_coef[idx] = result.params[1]
  idx += 1
  print(result.summary())

print('(a) all except chas, have small P-value, show the statistically significant association to \'crim\'.') 
#Question B
model = sm.OLS(response, sm.add_constant(predictor))
result = model.fit()
print(result.summary())
print('(b) zn, dis, rad, black, lstat, medv have small P-value which smaller than 0.05, can reject null hypothesis. ')
#Question C
mult_coef = result.params[1:]
plt.figure(1)
plt.xlabel('univariate regression coefficients')
plt.ylabel('multiple regression coefficients')
plt.plot(univariate_coef, mult_coef, 'o')
print('(c) see Figure 1')
#Question D
for column in predictor:
  x1 = predictor[column]
  x2 = x1 ** 2
  x3 = x1 ** 3
  predictor_poly = np.concatenate((x1.reshape(len(x1), 1), x2.reshape(len(x1), 1), x3.reshape(len(x1), 1)), axis=1)
  model = sm.OLS(response, sm.add_constant(predictor_poly))
  result = model.fit()
  print('{var} vs crim'.format(var=column))
  print(result.summary())
print('(d) plot the residual versus fitted check non-linear relationship.')
plt.show()
