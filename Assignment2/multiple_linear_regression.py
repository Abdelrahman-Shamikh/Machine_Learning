import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from Pre_processing import *


#Load players data
data = pd.read_csv('fifa19.csv')

#Deal with missing values
print(data.isna().sum())
#print(data.info())
data["Crossing"].fillna(data["Crossing"].mean(),inplace=True)
#data.fillna(0,downcast='infer')
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)
print(data.isna().sum())
X=data.iloc[:,1:40] #Features
Y=data['Value'] #Label

#Feature Encoding
cols=('Nationality','Club','Position')
X=Feature_Encoder(X,cols)

#Apply Standardization
"""scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)"""

#OR Normalization
X = featureScaling(X,0,1)

#Apply Linear Regression on the selected features
cls = linear_model.LinearRegression()
cls.fit(X,Y)
prediction= cls.predict(X)

print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), prediction))