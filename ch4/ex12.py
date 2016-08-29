import matplotlib.pyplot as plt
import numpy as np

#Question A
def Power():
  return 2 ** 3
print('(a) {result}'.format(result=Power()))

#Question B
def Power2(x, a):
  return x ** a
print('(b) {result}'.format(result=Power2(3, 8)))

#Question C
print('(c) 10^3={first}, 8^17={second}, 131^3={third}'.format(first=Power2(10, 3), second=Power2(8, 17), third=Power2(131, 3)))

#Question D
#Due to the difference between R and Python, using result() instead of return(result) here
#But I think it should be the same from functional perspective
def Power3(x, a):
  return lambda :Power2(x, a)

result = Power3(3, 8)
print('(d) result={result}'.format(result=result()))

#Question E
x = np.arange(10)
y = Power3(x, 2)
plt.plot(x, y())
plt.yscale('log')
plt.xscale('log')
plt.title('log scale of y=x^2')

#Question F
def PlotPower(x, a):
  plt.figure()
  x = np.arange(x)
  y = Power3(x, 3) 
  plt.plot(x, y())  
  plt.title(' y = x^3')
PlotPower(10, 3)

plt.show()
