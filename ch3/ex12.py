import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


print('Exercise 12, Answer:')
print('(a) The (3.38) show the numerator is x*y, the denominator is square y. So the beta hat\
       should be same when the square y equals square of x')
print('(b) ')
np.random.seed(1)
X = np.random.rand(100)
Y = 2 * X + np.random.rand(100)
df = pd.DataFrame({'x':X, 'y':Y})
model1 = smf.ols(formula='y~x + 0', data=df)
model2 = smf.ols(formula='x~y + 0', data=df)
res = model1.fit()
print(res.summary())
res = model2.fit()
print(res.summary())

print('(c) ')
Y = X
print(X)
np.random.shuffle(X)
print(sum(Y**2))
print(sum(X**2))
df = pd.DataFrame({'x':X, 'y':Y})
model1 = smf.ols(formula='y~x + 0', data=df)
model2 = smf.ols(formula='x~y + 0', data=df)
res = model1.fit()
print(res.summary())
res = model2.fit()
print(res.summary())

