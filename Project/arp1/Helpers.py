from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class CounterEncoder:
  def transform(self, X):
    ret = X.copy()
    ret = ret.fillna(0).apply(lambda x: 0 if x == 0 else len(str(x).split(',')))
    return ret

class MeansAndMods:
  def __init__(self):
    self.special = {}  #
  def fit(self, data):

    for col in data.columns:
      if data[col].dtype == 'object':  # Check if the column is categorical
        self.special[col] = data[col].mode()[0]
      elif data[col].dtype == 'float64':
        self.special[col] = data[col].mean()
      else:
        self.special[col] = int(data[col].mean())

  def transform(self, col, value=None):
    if value is None:
      return col.fillna(self.special[col.name])
    else:
      return col.apply(lambda x: self.special[col.name] if x == value else x)

# Note: The following line is Required as we want these 2 cols without any nulls

class KNN_PP:
  def __init__(self):
    self.knn =  KNeighborsClassifier(n_neighbors=1)

  def fit(self, X, Y):
    self.X_train = X[Y.notnull()]
    self.Y_train = Y.dropna()
    self.knn.fit(self.X_train, self.Y_train)

  def transform(self, X, Y):
    X_test = X[Y.isnull()]
    Y_pred = Y.copy()
    Y_pred[X_test.index] = self.knn.predict(X_test)
    return Y_pred

  def fit_transform(self, X, Y):
    self.fit(X, Y)
    return self.transform(X, Y)

class OHE:
  def __init__(self, removeCol):
    self.removeCol = removeCol
  def fit(self, data, suff = 'allowed'):
    self.addCols = set()
    for val in data[self.removeCol].value_counts().index:
      for string in str(val).split(','):
        self.addCols.add(string.lower() + '_' + suff)

  def transform(self, data):
    data[self.removeCol].fillna('', inplace=True)
    for col in self.addCols:
      data.insert(data.columns.get_loc(self.removeCol), col, np.zeros(data.shape[0], dtype=int))
      data.loc[data[self.removeCol].str.contains(col[:col.find('_')], case=False), col] = 1
    data.drop(columns=self.removeCol, inplace=True)

  def fit_transform(self, data, suff = 'allowed'):
    self.fit(data, suff)
    self.transform(data)