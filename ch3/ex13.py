import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt

np.random.seed(1)
#numpy.random.randn() will return sample from the standard normal distribution
#for N(u, sigma^2), use: sigma * np.random.randn(...) + u
#Question A
x = np.random.randn(100)
print(' Exercise 13, Answers:')
print('(a) ')
print(x)
#Question B
esp = 0.5 * np.random.randn(100)
print('(b) ')
print(esp)
#Question C
y = 0.5 * x - 1 + esp
print('(c) The length of vector y should be 100. Beta0 is -1, Beta1 is 0.5 .')
print(y)
#Question D
print('(d) see Figure1')
plt.figure(1)
plt.title('relationship between x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y, 'o')
#Question E
print('(e) ')
#add intercept
x_const = sm.add_constant(x)
#generate model
model = sm.OLS(y, x_const)
result = model.fit()
print(result.summary())
print(' The R-squared is 0.522 shows the model has something to improve.\
The small P-value reject null hypothesis of Beta. The Beta0_hat is {beta0}, Beta1_hat is\
 {beta1}'.format(beta0 = result.params[0], beta1 = result.params[1]))
#Question F
print('(f) see Figure1')
plt.plot(x, result.fittedvalues, '-', label='fitted')
#population regression line is the E(Y)=Beta0 + Beta1*X, becasue the mean of esp is 0. 
y_pop = 0.5 * x - 1
plt.plot(x, y_pop, '-', label='population line')
plt.legend()
#Question G
print('(g) ')
x_squr = x**2
x_poly = np.concatenate((x_const, x_squr.reshape(100, 1)), axis=1)
model = sm.OLS(y, x_poly)
result = model.fit()
print(result.summary())
print('The X^2 didn\'t increase the model fit, and the coefficient of X^2 has large P-value, the quadratic term has no relationship with the y.')
#Question H & I
print('(h) & (i), ignore')
#Question J
print('(j) The confidence interval widen as the noise increases.That\'s means, it\'s more predictable with less noise.')
plt.show()
