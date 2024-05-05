import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

#Load players data
data = pd.read_csv('fifa19.csv')
#Drop the rows that contain missing values
data.dropna(how ='any',inplace=True)
fifa_data=data.iloc[:,:]
X=data.iloc[:,7:10] #Features
Y=data['Value'] #Label

#built-in method
cls = linear_model.LinearRegression()
cls.fit(X,Y)
prediction= cls.predict(X)

#from-scratch method
L = 0.0000001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent
m1=0
m2=0
m3=0
c=0
X = np.array(X)
#Student code: initalize n with the number of players in your dataset
n = None
for i in range(epochs):
    #Student code
	# Implement gradient descent algorithm
	#Student code

# Student code
prediction_GD=None

print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), prediction))
print('Mean Square Error GD', metrics.mean_squared_error(np.asarray(Y), prediction_GD))