import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

fileName = r'../dataSet/Default.csv'
df = pd.read_csv(fileName).drop('Unnamed: 0', axis=1)
df['default'] = pd.Categorical(df['default']).codes
#Exercise 6
print('Exercise 6 ')

#Question A
print('(a) ')
model = smf.logit(formula='default~income+balance', data=df)
res = model.fit()
print(res.summary())
print((res.params))

#Question B
# boot_fn() is used for select sample from bootstrap sample set, then fit the sample
print('(b) ')
def boot_fn(dataSet, indexArray):
  frame = dataSet.iloc[indexArray, :]
  logit_model = smf.logit(formula='default~income+balance', data=frame)
  res = logit_model.fit()
  return res.params
params = boot_fn(df, np.random.randint(10000, size=10000))
#Question C
#I didn't find bootstrap function that can be used as boot.fn() and boot() in R
#I have to write them in python
# boot() is used for run bootstrap procedure
# boot() step i. random generates slices index that used for bootstrap
# boot() step ii. run function use generates slices index as parameter, total run R times
# boot() step iii. summary all of R results of step ii.
print('(c) ')
def boot(data, function, R):
  maxIndex = data.shape[0] 
  result = pd.DataFrame()
  for loop in range(R):
    randomIndex = np.random.randint(maxIndex, size=maxIndex)
    coef = function(data, randomIndex)
    #For DataFrames which don’t have a meaningful index,
    # you may wish to append them and ignore the fact that they may have overlapping indexes
    result = result.append(coef, ignore_index=True)
  return result

result = boot(df, boot_fn, 100)
incomeStd = np.std(result['income'])
balanceStd = np.std(result['balance'])
print('income standard deviation is {incomeStd}'.format(incomeStd=incomeStd))
print('balance standard deviation is {balanceStd}'.format(balanceStd=balanceStd))

#Question D
print('(d) The standard deviation of bootstrap has same magnitude as glm().')
