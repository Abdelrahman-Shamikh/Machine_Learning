import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from Pre_processing import *

#Load players data
data = pd.read_csv('fifa19.csv')
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)
fifa_data=data.iloc[:,:]
X=data.iloc[:,1:10] #Features
Y=data['Value'] #Label
cols=('Nationality','Club','Position')
X=Feature_Encoder(X,cols)

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)


model_1_poly_features = PolynomialFeatures(degree=2)
# transforms the existing features to higher degree features.
X_train_poly_model_1 = model_1_poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
poly_model1 = linear_model.LinearRegression()
scores = cross_val_score(poly_model1, X_train_poly_model_1, y_train, scoring='neg_mean_squared_error', cv=5)
model_1_score = abs(scores.mean())
poly_model1.fit(X_train_poly_model_1, y_train)
print("model 1 cross validation score is "+ str(model_1_score))

model_2_poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.
X_train_poly_model_2 = model_2_poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
poly_model2 = linear_model.LinearRegression()
scores = cross_val_score(poly_model2, X_train_poly_model_2, y_train, scoring='neg_mean_squared_error', cv=5)
model_2_score = abs(scores.mean())
poly_model2.fit(X_train_poly_model_2, y_train)

print("model 2 cross validation score is "+ str(model_2_score))

# predicting on test data-set
prediction = poly_model1.predict(model_1_poly_features.fit_transform(X_test))
print('Model 1 Test Mean Square Error', metrics.mean_squared_error(y_test, prediction))

# predicting on test data-set
prediction = poly_model2.predict(model_2_poly_features.fit_transform(X_test))
print('Model 2 Test Mean Square Error', metrics.mean_squared_error(y_test, prediction))
