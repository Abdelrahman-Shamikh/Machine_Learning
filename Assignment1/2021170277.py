import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import metrics
def compute_cost(prediction, y, w, b):

    total_cost=np.mean((prediction-y)**2)
    return total_cost


#fuction for gradient descent
def compute_gradient(x, y, m, c,L,epochs):
  n = x.shape[0]
  m = 0
  c = 0
  costs=[]
  for i in range(epochs):
      Y_pred = m*X + c  # The current predicted value of Y
      D_m = (-2/n) * sum((Y - Y_pred)* X)  # Derivative wrt m
      D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
      m = m - L * D_m  # Update m
      c = c - L * D_c
      cost =  compute_cost(Y_pred,y,m,c)
      costs.append(cost)
      # if i% math.ceil(epochs/10) == 0:
      #   print(f"Iteration {i}: Cost {cost} ",
      #   f"dm: {D_m}, dc: {D_c}  ",
      #   f"m: {m}, c:{c}")
  # Plotting epochs versus cost
  fig, (ax2) = plt.subplots(constrained_layout=True, figsize=(5, 4))
  ax2.plot(100 + np.arange(len(costs[100:])), costs[100:])
  ax2.set_title("Cost vs. iteration (tail)")
  ax2.set_ylabel('Cost')
  ax2.set_xlabel('iteration step')
  plt.show()
  return m,c
#Loading data
data = pd.read_csv('./assignment1dataset.csv')
Y=data['Performance Index']
X=data['Hours Studied']
n = float(len(X)) # Number of elements in X
# 1st model
m1, c1 = compute_gradient(X, Y, 0, 0, 0.01, 800)#from plot we can see since 800 it decreases alittle
prediction1 = m1 * X + c1
cost1 = compute_cost(prediction1, Y, m1, c1)
plt.scatter(X, Y)
plt.xlabel('Hours Studied', fontsize=20)
plt.ylabel('Performance Index', fontsize=20)
plt.plot(X, prediction1, color='red', linewidth=3)
plt.show()
print('Mean Square Error of 1st Model', metrics.mean_squared_error(Y, prediction1))
print('Cost Of 1st Model',cost1)

# # 2nd model

X=data['Sample Question Papers Practiced']
m2, c2 = compute_gradient(X, Y, 0, 0, 0.01, 800)# we see it is since 800 is constant
prediction2 = m2 * X + c2
cost2 = compute_cost(prediction2, Y, m2, c2)
plt.scatter(X, Y)
plt.xlabel('Sample Question Papers Practiced', fontsize=20)
plt.ylabel('Performance Index', fontsize=20)
plt.plot(X, prediction2, color='red', linewidth=3)
plt.show()
print('Mean Square Error of 2nd Model', metrics.mean_squared_error(Y, prediction2))
print('Cost Of 2nd Model',cost2)

# 3rd model
# if it took alot of time you can comment other models and run this only. using higher leaning rate will lead to NAN Values.
X=data['Previous Scores']
m3, c3 = compute_gradient(X, Y, 0, 0, 0.0001, 100000) #takes a lot of time (5 minutes on my machine)but as the graph shows from Pdf it starts to be fixed error from 100000
prediction3 = m3 * X + c3
cost3 = compute_cost(prediction3, Y, m3, c3)
plt.scatter(X, Y)
plt.xlabel('Previous Scores', fontsize=20)
plt.ylabel('Performance Index', fontsize=20)
plt.plot(X, prediction3, color='red', linewidth=3)
plt.show()
print('Mean Square Error of 3rd Model', metrics.mean_squared_error(Y, prediction3))
print('Cost Of 3rd Model',cost3)

# 4th model
X=data['Sleep Hours']
m4, c4 = compute_gradient(X, Y, 0, 0, 0.01, 2500)#2500 from graph try 3000 to know
prediction4 = m4 * X + c4
cost4 = compute_cost(prediction4, Y, m4, c4)
plt.scatter(X, Y)
plt.xlabel('Sleep Hours', fontsize=20)
plt.ylabel('Performance Index', fontsize=20)
plt.plot(X, prediction4, color='red', linewidth=3)
plt.show()
print('Mean Square Error of 4th Model', metrics.mean_squared_error(Y, prediction4))
print('Cost Of 4th Model',cost4)

# #5th model (combining)
# if it took alot of time you can comment other models and run this only 
X=data['Previous Scores']
z=data['Hours Studied']
X=X+z
m5, c5 = compute_gradient(X, Y, 0, 0, 0.0001, 150000) #takes a lot of time(6 minutes on my machine) 
prediction5 = m5 * X + c5
cost5 = compute_cost(prediction5, Y, m5, c5)
plt.scatter(X, Y)
plt.xlabel('Previous Scores and Hours Studied', fontsize=20)
plt.ylabel('Performance Index', fontsize=20)
plt.plot(X, prediction5, color='red', linewidth=3)
plt.show()
print('Mean Square Error of 5th Model', metrics.mean_squared_error(Y, prediction5))
print('Cost Of 5th Model',cost5)