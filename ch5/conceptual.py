

print('Excercise 1: use rules: Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y) if X, Y are independent.')
print('Excercise 2: ')
print('(a). 1-1/n')
print('(b). 1-1/n')
print('(c). each observation has same 1/n chance of equal the jth observation. so, for n sample, jth observation not in the sample should be (1-1/n)^n')
print('(d). 1-(1-1/5)^5=0.672')
print('(e). 1-(1-1/100)^10=0.634')
print('(f). 1-(1-1/10000)^10000=0.632')
print('(g) and (h) ignore')
print('Exercise 3: ')
print('(a) Randomly dividing the set of observation into k groups, each group is a fold. At beginging, 1st fold as validation set, others are training set. The mean squared error, MSE is computed on the observations. This procedure is repeated k times.')
print('(b) i. k-fold has more precise test error rate than validation approach. But k-fold cost much on computation.')
print('(b) ii. k-fold has perform better computation than LOOCV and involve the bias-variance trade-off. ')
print('Exercise 4: Bootstrap can be used to estimate the statndard errors. Bootstrap can get more precise standard error, standard deviation can derive by SE * squr(n).')