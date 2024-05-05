
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("./assignment2dataset.csv")
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)

def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

data=Feature_Encoder(data,["Extracurricular Activities"])
def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X
a=featureScaling(data[['Hours Studied','Previous Scores','Sleep Hours','Performance Index']],0,1)

data1=pd.DataFrame(a)
data['Hours Studied']=data1.iloc[:,0]
data['Previous Scores']=data1.iloc[:,1]
data['Sleep Hours']=data1.iloc[:,2]
data['Performance Index']=data1.iloc[:,-1]
X=data[['Hours Studied','Previous Scores','Sleep Hours']]
Y=data['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)


def polynomial_regression(X, degree):
    data2 = pd.DataFrame()
    dic = {}

    # Generate polynomial features up to the specified degree
    for i in range(1, degree + 1):
        for col in X.columns:
            data2[f'{col} power {i}'] = X[col] ** i
            if col not in dic:
                dic[col] = {}
            dic[col][i] = data2[f'{col} power {i}']

    a = list(range(1, degree + 1))


    for i, col_i in zip(range(len(X.columns)), X.columns):
        print('i=',i)
        print('col',col_i)
        for fpowers in a:
            for j, col_j in zip(range(i + 1, len(X.columns)), X.columns[i + 1:]):
                print('j',j)
                print('jcol=',col_j)
                for spowers in a:
                    print(fpowers)
                    print(spowers)
                    if fpowers + spowers <= degree:
                        data2[f'{col_i}, {col_j} power {fpowers}, {spowers}'] = dic[col_i][fpowers] * dic[col_j][
                            spowers]

    data2.insert(0, 'Ones', 1)

    return data2
deg=input("enter the degree ")
# transforms the existing features to higher degree features.
X_train_poly=polynomial_regression(X_train,int(deg))

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
X_test_poly=polynomial_regression(X_test,int(deg))
# predicting on test data-set
prediction = poly_model.predict(X_test_poly)
print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('training Mean Square Error', metrics.mean_squared_error(y_train, y_train_predicted))
print('testing Mean Square Error', metrics.mean_squared_error(y_test, prediction))
true_player_value=np.asarray(y_test)[0]
predicted_player_value=prediction[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
