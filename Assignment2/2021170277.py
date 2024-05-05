import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from itertools import combinations,permutations

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import product
from itertools import combinations_with_replacement

def Feature_Encoder(X,col):
    lbl = LabelEncoder()
    lbl.fit(list(X[col].values))
    X[col] = lbl.transform(list(X[col].values))
    return X


def generate_polynomial_features(X, degree):
    nexamples, ncols = X.shape
    trans = []
    trans.append(np.ones((nexamples, 1)))
    for d in range(1, degree + 1):
      for i in product(range(ncols),repeat= d):
          transed = np.prod(X.iloc[:, list(i)], axis=1)
          trans.append(transed)
    return np.column_stack(trans)


data = pd.read_csv('./assignment2dataset.csv')
Y=data['Performance Index']
data = Feature_Encoder(data,'Extracurricular Activities')

corr = data.corr()
top_feature = corr.index[abs(corr['Performance Index'])>0.045]
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)

X=data[top_feature]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)
o=int(input("please enter degree "))
poly = generate_polynomial_features(X_train, o)
poly_model = linear_model.LinearRegression()
poly_model.fit(poly, y_train)

y_train_predicted = poly_model.predict(poly)
ypred=poly_model.predict(generate_polynomial_features(X_test,o))
print('training Mean Square Error', metrics.mean_squared_error(y_train, y_train_predicted))
prediction = poly_model.predict(generate_polynomial_features(X_test,o))
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
