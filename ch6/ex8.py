import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split

#Exercise 8
print('Exercise 8 ')
# copy from http://planspace.org/20150423-forward_selection_with_statsmodels/
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = list(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

def backward_selected(data, response):
    """Linear model designed by backward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by backward selection
           evaluated by adjusted R-squared
    """
    remaining = list(data.columns)
    remaining.remove(response)
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            selected = list(remaining)
            selected.remove(candidate)
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(remaining))
    model = smf.ols(formula, data).fit()
    return model

#Question A
print('(a) ')
np.random.seed(2)
X = np.random.randn(100)
noise = np.random.randn(100)
#Question B
print('(b) ')
beta0 = 2
beta1 = 5
beta2 = 2
beta3 = -1
Y = beta0 + beta1 * X + beta2 * (X ** 2) + beta3 * (X ** 3) + noise

#Question C
print('(c) ')
dataSet = pd.DataFrame({'x':X, 'x_2':X**2, 'x_3':X**3, 'x_4':X**4, 'x_5':X**5, 'x_6':X**6, 'x_7':X**7, 'x_8':X**8, 'x_9':X**9, 'x_10':X**10, 'y':Y})
model = forward_selected(dataSet, 'y')
print(model.model.formula)
print('for adjust-Rsquared, 3 predictors perform best. The result is strong related with the choice of betaX. ')

#Question D
print('(d) ')
model = backward_selected(dataSet, 'y')
print(model.model.formula)
print('for adjust-Rsquared, 5 predictors perform best, quiet different with forward stepwise. ')
#Question E
print('(e) ')
trainSet, validationSet = train_test_split(dataSet, train_size=0.5)
def LassoLambda(alpha, trainSet, validationSet):
  lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
  lassoreg.fit(trainSet.loc[:, 'x':'x_10'], trainSet.loc[:, 'y'])
  predict_lasso = lassoreg.predict(validationSet.loc[:, 'x':'x_10'].values)
  error_rate = np.linalg.norm(predict_lasso - validationSet.loc[:, 'y'], ord=2)
  return error_rate
alphaList = list(np.arange(0.01, 1, 0.001))
errorRateList = []
for alpha in alphaList:
  error_rate = LassoLambda(alpha, trainSet, validationSet)
  errorRateList.append(error_rate)
plt.plot(alphaList, errorRateList, 'o')

#Question F
print('(f) ignore.')


plt.show()
