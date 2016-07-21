import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


np.random.seed(1)
x1 = np.random.uniform(size=100)
x2 = 0.5 * x1 + np.random.randn(100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + np.random.randn(100)
X_raw = np.concatenate((x2.reshape(100, 1), x1.reshape(100, 1)), axis=1)
X = sm.add_constant(X_raw)
model = sm.OLS(y, X)
result = model.fit()
print(result.summary())
print('Exercise 14, Answer:')
#Question A
print('(a) The form of linear model is Y = Beta0 + Beta1 * X1 + Beta2 * X2. The regression\
coefficents is Beta0={beta0}, Beta1={beta1}, Beta2={beta2}'.format(beta0=result.params[0],\
beta1=result.params[1], beta2=result.params[2]))
corr = np.corrcoef(x1, x2)
#Question B
plt.figure(1)
plt.title('relationship between x1 and x2')
plt.plot(x1, x2, 'o')
print('(b) The correlation between x1 and x2 is {corr}. Scatterplot see Figure1'.format(corr=corr[0,1]))
#Question C
print('(c) The regression coefficents is Beta0={beta0}, Beta1={beta1}, Beta2={beta2}. The P-value of Beta1 smaller than 0.05, but Beta2\'s P-value is large. So we can reject null hypothesis to Beta1 only.'.format(beta0=result.params[0], beta1=result.params[1], beta2=result.params[2]))
#Question D
model = sm.OLS(y, x1)
result = model.fit()
print(result.summary())
print('(d) The regression result shows the strong relation ship between y and x1. It can conclude that collinearity will impact the regression result. The collinearity corrode the regression model. And the small P-value reject the null hypothesis of Beta1=0.')
#Question E
model = sm.OLS(y, x2)
result = model.fit()
print(result.summary())
print('(e) We can get almost same result as Question D.')
#Question F
print('(f) The result of (c)-(e) don\'t contradict each other. The results show collinear will collapse the linear model. And we can use correlation matrix to detect correlationship among predictors.')
#Question G
print('(g) ignore.')
plt.show()
