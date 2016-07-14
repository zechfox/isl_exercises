import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats

fileName = r'../dataSet/Carseats.csv'
df = pd.read_csv(fileName)
X_raw = pd.DataFrame()
#statsmodels handle numeric object, has to categorical object then numeric it
X_raw['Price'] = pd.Categorical(df['Price']).codes
X_raw['Urban'] = pd.Categorical(df['Urban']).codes
X_raw['US'] = pd.Categorical(df['US']).codes

y = df['Sales']
print(df['US'])

X = sm.add_constant(X_raw)
est = sm.OLS(y, X).fit()
print('(a) ')
print(est.summary())

print('(b) The F-statistic show the relationship between predictor and response. Price and Urban show negative relationship to Sales while US show positve relationship. The large P-value support null-hypersis of Urban, that\'s means Urban has less expression in Price.')
print('(c) Sales = 9.6291 + (-0.578) * Price + (-0.0055) * Urban + 1.1703 * US')
print('(d) We can reject null hypothesis for predictor: Price and US due to low P-value and t-statistic.')
#exclude Urban because no evidence of association with the response
X_raw = X_raw.drop('Urban', axis=1)
X = sm.add_constant(X_raw)
est = sm.OLS(y, X).fit()
print('(e)')
print(est.summary())
print('(f) The R-squared almost same in (a) and (e). ')
print('(g) The 95% confidence can be obtain in the summary')
print('(h) No potential outliers are suggested.')
