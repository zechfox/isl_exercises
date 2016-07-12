import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats


fileName = r'../dataSet/Auto.csv'
#if 'coerce', then invalid parsing will be set as NaN
df = pd.read_csv(fileName)
df_numeric = df.apply(pd.to_numeric, args=('coerce',))
mask = ~np.isnan(df_numeric['cylinders'].values) & ~np.isnan(df_numeric['displacement'].values)\
       & ~np.isnan(df_numeric['horsepower'].values) & ~np.isnan(df_numeric['weight'].values)\
       & ~np.isnan(df_numeric['acceleration'].values) & ~np.isnan(df_numeric['year'].values)\
       & ~np.isnan(df_numeric['origin'].values)
X_raw = df_numeric[['cylinders','displacement','horsepower','weight','acceleration','year','origin']][mask]
y = df_numeric['mpg'][mask]

X = sm.add_constant(X_raw)
est = sm.OLS(y,X).fit()
print('Exercise 9 Answer:')
print('(a) see figure 1')
pd.scatter_matrix(df, alpha=0.5)
print('(b) ')
#correlations = np.corrcoef(pd.concat([y, X_raw], axis=1), rowvar=0)
correlations = np.corrcoef(df_numeric.loc[:,'mpg':'origin'][mask], rowvar=0)
print('(c)')
print(est.summary())
print('(c) i. The null-hypersis of all the regression coefficients are zero can be reject by large F-statistic with very small P-value.')
print('(c) ii. From P-value of each predictor, all predictor has statistically significant relationship to the response except cylinders, horsepower and acceleration.')
print('(c) iii. The coefficient of year show positive relationship. And increase of 1 year gain 0.7508 increase of mpg. It\'s means cars become more fuel efficient by year.')
print('(d) see figure 2.')
plt.figure(2)
# R plot for lm object will generate 6 plots: residuals against fitted values, sqrt(|residuals|) against fitted values, Normal Q-Q plot,
#Cook's distances versus row lables, residuals against leverages, and Cook's distances against leverage. By default, the first 3 and 5 are provided
# we plot default by python   

#residuals vs fitted values
plt.subplot(221)
plt.title('Residual vs Fitted Value')
fitted_values = est.fittedvalues
residuals = est.resid
plt.xlabel('fitted values')
plt.ylabel('residuals')
plt.plot(fitted_values, residuals,'o')

#sqrt(residuals) vs fitted values
plt.subplot(222)
plt.title('sqrt(residuals vs fitted value')
#Normalized residual
residuals_std = stats.zscore(residuals)
residuals_sqrt = residuals_std ** .5
plt.xlabel('fitted values')
plt.ylabel('sqrt(residuals')
plt.plot(fitted_values, residuals_sqrt, 'o')

#Normal Q-Q
plt.subplot(223)
plt.title('Normal Q-Q')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Standardized residuals')
stats.probplot(residuals_std, dist="norm", plot=plt)

#residuals against leverages
plt.subplot(224)
plt.title('Residuals vs Leverages')
plt.xlabel('Leverages')
plt.ylabel('Residuals')
#H=XInverse(Transpose(X)X)Transpose(X)
X_raw_t = X_raw.transpose()
X_t_mult_X_inv = np.linalg.inv(np.matmul(X_raw_t, X_raw))
X_mult_X_t_mult_X_inv = np.matmul(X_raw, X_t_mult_X_inv)
leverages = np.diag(np.matmul(X_mult_X_t_mult_X_inv, X_raw_t))
plt.plot(leverages, residuals_std, 'o')
print('(d) The residual vs Fitted values show non-linear relationship between preditor and response. Some outlier exist and few infulential point show in residuals vs leverage, the infulential point show high leverage.')

print('(e) and (f) ignore')
plt.show()

