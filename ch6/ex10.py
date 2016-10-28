import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split

# copy from http://planspace.org/20150423-forward_selection_with_statsmodels/
def forward_selected(data, response):
  remaining = list(data.columns)
  remaining.remove(response)
  selected = []
  bestScoreList = []
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
    bestScoreList.append((len(data.columns) - 1 - len(remaining),best_new_score))
  formula = "{} ~ {} + 1".format(response,
                               ' + '.join(selected))
  model = smf.ols(formula, data).fit()
  return model, bestScoreList
#Exercise 10
print('Exercise 10 ')

#Question A
print('(a) ')
p = 20
n = 1000
np.random.seed(3)
X = pd.DataFrame(np.random.randn(n, p))
noise = pd.Series(np.random.randn(n))
BETA = pd.Series(np.random.randn(p))
BETA[4] = 0
BETA[7] = 0
BETA[12] = 0
BETA[16] = 0
Y = X.dot(BETA).add(noise)
dataSet = pd.concat([X, Y], axis=1, ignore_index=True)
dataSet.columns = ['x','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','Y']
#Question B
print('(b) ')
trainSet, validationSet = train_test_split(dataSet, train_size=0.1)
#Question C
print('(c) ')
model, bestScoreList = forward_selected(trainSet, 'Y')
plt.figure(1)
plt.plot(*zip(*bestScoreList))

#Question D
print('(d) ')
model, bestScoreList = forward_selected(validationSet, 'Y')
plt.figure(2)
plt.plot(*zip(*bestScoreList))

#Question E
print('(e) ')
print('model size of 16 perform best in test set')

#Question F
print('(f) ')
print(model.model.formula)
print('the predictor number of  best model generates by test set is same as true model.')

#Question G
print('(g) ')


plt.show()
