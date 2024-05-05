import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
def Feature_Encoder(X,col):
    lbl = LabelEncoder()
    lbl.fit(list(X[col].values))
    X[col] = lbl.transform(list(X[col].values))
    return X
#Load players data
data = pd.read_csv('./assignment2dataset.csv')
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)

Y=data['Performance Index'] #Label
Feature_Encoder(data,'Extracurricular Activities')
X=data.iloc[:,0:5] #Features

#Feature Selection
#Get the correlation between the features
corr = data.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Performance Index'])>0.045]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
X = X[top_feature]

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.50,shuffle=True)

poly_features = PolynomialFeatures(degree=3)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred=poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('training Mean Square Error', metrics.mean_squared_error(y_train, y_train_predicted))


print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

true_player_value=np.asarray(y_test)[0]
predicted_player_value=prediction[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))