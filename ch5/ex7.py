import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

fileName = r'../dataSet/Weekly.csv'
df_raw = pd.read_csv(fileName).drop('Unnamed: 0', axis=1)
print(df_raw.head(5))
df_raw['Direction'] = pd.Categorical(df_raw['Direction']).codes
df = df_raw

#Exercise 7
print('Exercise 7 ')

#Question A
print('(a) ')
model = smf.glm(formula='Direction~Lag1+Lag2', data=df)
result = model.fit()
print(result.summary())

#Question B
print('(b) ')
df = df_raw.drop(0, axis=0)
model = smf.glm(formula='Direction~Lag1+Lag2', data=df)
result = model.fit()
print(result.summary())

#Question C
print('(c) ')
predict = result.predict(df_raw.iloc[0, :])
print('predict Direction is {predict_direction}, actual Direction is {actual_direction}'.format(predict_direction= 1 if(predict>0.5) else 0, actual_direction=df_raw['Direction'][0]))

#Question D
print('(d) ')
length = df_raw.shape[0]
prediction = pd.Series(np.zeros(length))
for i in range(length):
  df = df_raw.drop(i, axis=0)
  model = smf.glm(formula='Direction~Lag1+Lag2', data=df)
  result = model.fit()
  prediction[i] = 1 if(result.predict(df_raw.iloc[i, :])>0.5) else 0

print('total correct prediction is {sum}'.format(sum=np.sum(prediction)))   

#Question E
print('(e) ')
print('average correct rate is {rate}'.format(rate=np.average(prediction)))
