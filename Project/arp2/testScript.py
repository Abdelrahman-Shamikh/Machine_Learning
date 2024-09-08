import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import kendalltau
import time
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
import pickle
from Helpers import OHE
from Helpers import KNN_PP
from Helpers import MeansAndMods
from Helpers import CounterEncoder

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

data = pd.read_csv('ApartmentRentPrediction_Milestone2.csv')


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

data['RentCategory'] = data['RentCategory'].replace({'Medium-Priced Rent': 1, 'Low Rent': 0, 'High Rent': 2})

cols=('category','title','body','cityname', 'state', 'source', 'address','fee','currency','price_type')

lbls = []
for lbl in loaded_lbls:
    lbls.append(lbl)

for c in cols:
  lbl = lbls.pop(0)
  for idx, val in data[c].items():
    if val not in lbl.classes_:
      data.at[idx,c]= loaded_mam.special[c]
  data[c] = lbl.transform(data[c])

RentCategoryData = data['RentCategory']
data.drop(columns=['RentCategory'], inplace = True)

column_names = data.columns

data.info()
data_scaled = loaded_scaler.transform(data)

data_scaled_df = pd.DataFrame(data_scaled, columns=column_names)

data = data_scaled_df
data['RentCategory'] = RentCategoryData.values

y = data['RentCategory']
x_rf = data[['state','yes_photo','cats_allowed','dogs_allowed','cityname','square_feet','bathrooms','bedrooms','longitude','time']]


y_pred = loaded_rf.predict(x_rf)
print("Random Forest accuracy:", accuracy_score(y, y_pred))