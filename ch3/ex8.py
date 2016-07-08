import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

fileName = r'../dataSet/Auto.csv'
#if 'coerce', then invalid parsing will be set as NaN
df = pd.read_csv(fileName).apply(pd.to_numeric, args=('coerce',))
#exclude nan, nan will cause linregress out put nan
mask = ~np.isnan(df['horsepower'].values) & ~np.isnan(df['mpg'].values)
#fancy indexing for 'horsepower' and 'mpg'
#use boolean array to select the corresponding elements of another array
predictor = df['horsepower'].values[mask]
response = df['mpg'].values[mask]
slope, intercept, r_value, p_value, std_err = stats.linregress(predictor, response)

print('Exercise 8 answer:')
print('(a)')
print('slope is',slope)
print('intercept is',intercept)
#to speedup calculation, use math.pow()
print('R-squared is', r_value**2)
print('P-value is',p_value)

print('(a) i. the p-value show the rejection on null-hypersis. We can conclude the predictor and response has relationship.')
print('(a) ii. the R-squared measures the proportion of variablility in response that can be explaned using predictor. So, about 60.59% response can be explained by predictor.')
print('(a) iii. the slope is positive, the relationship  between the predictor and response is positive.')
print('(b) see figure 1')
predict_mpg = intercept + predictor * slope

plt.figure(1)
plt.plot(predictor, response, 'o')
plt.plot(predictor, predict_mpg, 'k-')

# R plot for lm object will generate 6 plots: residuals against fitted values, sqrt(|residuals|) against fitted values, Normal Q-Q plot, 
#Cook's distances versus row lables, residuals against leverages, and Cook's distances against leverage. By default, the first 3 and 5 are provided
# we plot default by python 
print('(c) see figure 2')
plt.figure(2)
#residuals against fitted values
plt.subplot(221)
plt.title('residuals vs fitted')
residual = response - predict_mpg
plt.ylabel('residuals')
plt.xlabel('fitted')
plt.plot(predict_mpg, residual, 'o')
#sqrt(|residuals|) against fitted values
plt.subplot(222)
plt.title('sqr(resituals) vs fitted')
plt.ylabel('sqrt(residuals)')
plt.xlabel('fitted')
residual_sqrt = residual ** 0.5
plt.plot(predict_mpg, residual_sqrt, 'o')

#Normal Q-Q
plt.subplot(223)
#standardized by zscore
residual_std = stats.zscore(residual)
stats.probplot(residual_std, dist="norm", plot=plt)
#residuals against leverages
plt.subplot(224)
predictor_variance = np.var(predictor)
predictor_mean = np.mean(predictor)
leverage = [(1/predictor.size + (xi - predictor_mean)**2 / (predictor.size * predictor_mean)) for xi in predictor]
plt.title('resituals vs leverage')
plt.ylabel('residuals')
plt.xlabel('leverage')
plt.plot(leverage, residual, 'o')

print('(c) the residuals vs fitted shows the residual and fitted are not random, but with non-linear relation.')
print('    And the sqrt(residuals) vs fitted shows the non-linear evidance more significant')

plt.show()
