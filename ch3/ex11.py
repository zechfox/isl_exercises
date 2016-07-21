import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


#pseudo-random numbers work by starting with a number(the seed), 
#the resulting number is then used as the seed to generate the next 'random' number.
#When you set the seed, it does the same thing ever time, giving you the same numbers.
#If you want seemly random numbers, do not set the seed.
#NOTE: numpy.random.seed(), the main difficulty is not thread-safe. So-far, random.random.seed() is thread-safe. 
np.random.seed(1)
X = np.random.randn(100)
Y = 2 * X + np.random.randn(100)
df = pd.DataFrame({'x':X, 'y':Y})
model = smf.ols(formula='y~x + 0', data=df)
res = model.fit()

print('Exercise 11, Answer:')
print('(a) The coefficient estimated is {coef}, the standard error of the coefficient is {std_err}, \
       the t-statistic is {t_stat}, and P-value is {p_value}. The small P-value can reject null-hypothesis'\
       .format(coef = res.bse.values, std_err = res.params.values, t_stat = 'None', p_value = res.f_pvalue))
model2 = smf.ols(formula='x~y + 0', data=df)
res = model2.fit()
print('(b) The coefficient estimated is {coef}, the standard error of the coefficient is {std_err}, \
       the t-statistic is {t_stat}, and P-value is {p_value}. The small P-value can reject null-hypothesis'\
       .format(coef = res.bse.values, std_err = res.params.values, t_stat = 'None', p_value = res.f_pvalue))
print('(c) The coefficient almost reverse between (a) and (b), and has same statistic result')
print('(d),(e),(f) igore')

print(res.summary())
