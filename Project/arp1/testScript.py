
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, preprocessing, ensemble, model_selection, feature_selection
from scipy.stats import boxcox
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from Helpers import *

import pickle

with open('testScript.pkl', 'rb') as f:
    loaded_rf = pickle.load(f)
    loaded_ce = pickle.load(f)
    loaded_mam = pickle.load(f)
    loaded_city_KP = pickle.load(f)
    loaded_state_KP = pickle.load(f)
    loaded_photo_ohe = pickle.load(f)
    loaded_pets_ohe = pickle.load(f)
    loaded_scaler = pickle.load(f)
    loaded_lbls = pickle.load(f)

data = pd.read_csv('ApartmentRentPrediction.csv')
data['price_display'] = data['price_display'].str.replace(r'[^0-9]', '', regex=True)
data['price_display'] = data['price_display'].astype(float)

data.drop(columns=['price'], inplace=True)


data['amenities'] = loaded_ce.transform(data['amenities'])


data['latitude'] = loaded_mam.transform(data['latitude'])
data['longitude'] = loaded_mam.transform(data['longitude'])

data['cityname'] = loaded_city_KP.transform(data.loc[:, ['latitude', 'longitude']], data['cityname'])
data['state'] = loaded_state_KP.transform(data.loc[:, ['latitude', 'longitude']], data['state'])

loaded_pets_ohe.transform(data)

loaded_photo_ohe.transform(data)
data.drop(columns=['no_photo'],inplace = True)
for col in data.columns:
  data[col] = loaded_mam.transform(data[col])


# data['bathrooms'] = loaded_mam.transform(data['bathrooms'])
# data['bedrooms']  = loaded_mam.transform(data['bedrooms'])
# data['address'] = loaded_mam.transform(data['address'])

data['bathrooms'] = data['bathrooms'].astype(int)
data['bedrooms'] = data['bedrooms'].astype(int)

cols=('category','title','body','cityname', 'state', 'source', 'address','fee','currency','price_type')
for c in cols:
  lbl = loaded_lbls.pop(0)
  for idx, val in data[c].items():
    if val not in lbl.classes_:
      data.at[idx,c]= loaded_mam.special[c]
  data[c] = lbl.transform(data[c])

data.info()

price_t = data['price_display']
data.drop(columns=['price_display'], inplace = True)

column_names = data.columns


data_scaled = loaded_scaler.transform(data)

data_scaled_df = pd.DataFrame(data_scaled, columns=column_names)

data = data_scaled_df
data['price_display'] = price_t.values

y = data['price_display']
x_rf = data[['bathrooms','bedrooms','square_feet','state','longitude']]


y_pred = loaded_rf.predict(x_rf)


data_mse = metrics.mean_squared_error(y, y_pred)
r2 = r2_score(y,y_pred )

print('Random Forest data Mean Square Error:', data_mse)
print("Random Forest r2test score:", r2)