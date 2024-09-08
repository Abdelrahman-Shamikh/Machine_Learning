import pandas as pd
from sklearn.metrics import accuracy_score
import cloudpickle
from file import CounterEncoder
from file import KNN_PP
from file import MeansAndMods
from file import OHE
from file import mms
from sklearn.preprocessing import MinMaxScaler
import pickle

load_lbls = []

with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript.pkl', 'rb') as f:
    loaded_rf = pickle.load(f)
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript2.pkl', 'rb') as f:
    loaded_ce = pickle.load(f)
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript3.pkl', 'rb') as f:
    loaded_mam = pickle.load(f)
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript4.pkl', 'rb') as f:
    loaded_city_KP = pickle.load(f)
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript5.pkl', 'rb') as f:
    loaded_state_KP = pickle.load(f)
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript6.pkl', 'rb') as f:
    loaded_photo_ohe = pickle.load(f)
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript7.pkl', 'rb') as f:
    loaded_pets_ohe = pickle.load(f)
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript8.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)
with open('E:\\Abdelrahman\\Machine_Learning\\Assignments\\Project\\testScript9.pkl', 'rb') as f:
    load_lbls = pickle.load(f)
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

data['bathrooms'] = data['bathrooms'].astype(int)
data['bedrooms'] = data['bedrooms'].astype(int)
data['RentCategory'] = data['RentCategory'].replace({'Medium-Priced Rent': 1, 'Low Rent': 0, 'High Rent': 2})
cols=('category','title','body','cityname', 'state', 'source', 'address','fee','currency','price_type')
for c in cols:
  lbl = load_lbls.pop(0)
  for idx, val in data[c].items():
    if val not in lbl.classes_:
      data.at[idx,c]= loaded_mam.special[c]
  data[c] = lbl.transform(data[c])
RentCategoryData = data['RentCategory']
print(data.columns)
data.drop(columns=['RentCategory'], inplace = True)
column_names = data.columns
print("info ")
data.info()
data_scaled = loaded_scaler.transform(data)
data_scaled_df = pd.DataFrame(data_scaled, columns=column_names)
data = data_scaled_df
data['RentCategory'] = RentCategoryData.values
y = data['RentCategory']
x_rf = data[['state','yes_photo','cats_allowed','dogs_allowed','cityname','square_feet','bathrooms','bedrooms','longitude','time']]
y_pred = loaded_rf.predict(x_rf)
print("Random Forest accuracy:", accuracy_score(y, y_pred))

