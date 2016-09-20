import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import LeaveOneOut

#Exercise 8
print('Exercise 8 ')

#Question A
print('(a) n = 100, p = 2')
X = np.random.randn(100)
Y = X - 2 * X ** 2 + np.random.randn(100)

#Question B
print('(b) see Figure 1')
plt.plot(X, Y, 'o')


#Question C
print('(c) ')
np.random.seed(1)
X = np.random.randn(100)
Y = X - 2 * X ** 2 + np.random.randn(100)
df_raw = pd.DataFrame({'x':X, 'x_2':X**2, 'x_3':X**3, 'x_4':X**4, 'y':Y})
#I hate sklearn, because terrible documents and example, and confusing parameter
#glm_model = LogisticRegressionCV(cv = LeaveOneOut(100))
#glm_model.fit(df['x'].reshape(-1, 1), df['y'])

prediction = pd.DataFrame(np.zeros((100, 4)))
for i in range(100):
  df = df_raw.drop(i, axis=0) 
  model1 = smf.ols(formula='y~x', data=df)
  result1 = model1.fit()
  model2 = smf.ols(formula='y~x + x_2', data=df)
  result2 = model2.fit()
  model3 = smf.ols(formula='y~x + x_2 + x_3', data=df)
  result3 = model3.fit()
  model4 = smf.ols(formula='y~x + x_2 + x_3 + x_4', data=df)
  result4 = model4.fit()
  prediction.iloc[i, 0] = 1 if(np.fabs(result1.predict(df_raw.iloc[i, :]) - df_raw['y'][i]) < 1) else 0 
  prediction.iloc[i, 1] = 1 if(np.fabs(result2.predict(df_raw.iloc[i, :]) - df_raw['y'][i]) < 1) else 0 
  prediction.iloc[i, 2] = 1 if(np.fabs(result3.predict(df_raw.iloc[i, :]) - df_raw['y'][i]) < 1) else 0 
  prediction.iloc[i, 3] = 1 if(np.fabs(result4.predict(df_raw.iloc[i, :]) - df_raw['y'][i]) < 1) else 0 

model = smf.ols(formula='y~x + x_2', data=df_raw)
result = model.fit()
predict = result.predict(df_raw)
print(result.summary())
plt.figure(2) 
plt.plot(predict, df_raw['y'], 'o')
print('i. total correct prediction is {sum}'.format(sum=np.sum(prediction.iloc[:,0])))
print('ii. total correct prediction is {sum}'.format(sum=np.sum(prediction.iloc[:, 1])))
print('iii. total correct prediction is {sum}'.format(sum=np.sum(prediction.iloc[:, 2])))
print('iv. total correct prediction is {sum}'.format(sum=np.sum(prediction.iloc[:, 3])))

#Question D
print('(d) ')
np.random.seed(3)
X = np.random.randn(100)
Y = X - 2 * X ** 2 + np.random.randn(100)
df_raw = pd.DataFrame({'x':X, 'x_2':X**2, 'x_3':X**3, 'x_4':X**4, 'y':Y})

prediction = pd.DataFrame(np.zeros((100, 4)))
for i in range(100):
  df = df_raw.drop(i, axis=0) 
  model1 = smf.ols(formula='y~x', data=df)
  result1 = model1.fit()
  model2 = smf.ols(formula='y~x + x_2', data=df)
  result2 = model2.fit()
  model3 = smf.ols(formula='y~x + x_2 + x_3', data=df)
  result3 = model3.fit()
  model4 = smf.ols(formula='y~x + x_2 + x_3 + x_4', data=df)
  result4 = model4.fit()
  prediction.iloc[i, 0] = 1 if(np.fabs(result1.predict(df_raw.iloc[i, :]) - df_raw['y'][i]) < 1) else 0 
  prediction.iloc[i, 1] = 1 if(np.fabs(result2.predict(df_raw.iloc[i, :]) - df_raw['y'][i]) < 1) else 0 
  prediction.iloc[i, 2] = 1 if(np.fabs(result3.predict(df_raw.iloc[i, :]) - df_raw['y'][i]) < 1) else 0 
  prediction.iloc[i, 3] = 1 if(np.fabs(result4.predict(df_raw.iloc[i, :]) - df_raw['y'][i]) < 1) else 0 

print('i. total correct prediction is {sum}'.format(sum=np.sum(prediction.iloc[:,0])))
print('ii. total correct prediction is {sum}'.format(sum=np.sum(prediction.iloc[:, 1])))
print('iii. total correct prediction is {sum}'.format(sum=np.sum(prediction.iloc[:, 2])))
print('iv. total correct prediction is {sum}'.format(sum=np.sum(prediction.iloc[:, 3])))

#Question E
print('(e) It seems model ii has the best performance overall. Because it almost same as the actual data pattern.')

#Question F
print('(f) ')
plt.show()

