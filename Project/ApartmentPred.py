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


"""read the data"""

data = pd.read_csv('ApartmentRentPrediction.csv')

print(data.info())
print(data.shape)

"""put the prediction column at the end"""

predicted_col = data['price_display']
data.drop(columns=['price_display', 'price'], inplace=True)
data['price_display'] = predicted_col

"""modify target format"""
data['price_display'] = data['price_display'].str.replace(r'[^0-9]', '', regex=True)
data['price_display'] = data['price_display'].astype(float)


print(data['price_display'].head(10))

"""number of nulls"""
for column in data.columns:
  print(column, '\t',data[column].isna().sum())

"""replace each row in amenities with the **count** of the amenities"""
data['amenities'] = data['amenities'].fillna(0).apply(lambda x: 0 if x == 0 else len(str(x).split(',')))

"""remove noise"""
print(data['category'].value_counts())
data = data.drop(data[(data['category'] == 'housing/rent/short_term') | (data['category'] == 'housing/rent/home')].index)
print(data['category'].value_counts())

data.dropna(subset=['latitude', 'longitude'], inplace=True)

#filling null values of columns cityname/state with nearthest cityname/state according to euclidean distance on columns ['latitude', 'longitude']

def K_nearthest_places_null_filter(targetNull, x, y, k):
  missing_data = data[data[targetNull].isna()]
  coordinates = set(zip(missing_data[x], missing_data[y]))

  for coord in coordinates:
      # Filter out rows with non-missing targetNull
      non_missing_data = data.dropna(subset=[targetNull])

      # Calculate Euclidean distances between the current coordinate and all non-missing coordinates
      distances = cdist([coord], non_missing_data[[x, y]], metric='euclidean')

      # Get indices of the k nearest points
      nearest_indices = np.argsort(distances)[0][:k]

      # Get the corresponding targetNull values
      nearest_cities = non_missing_data.iloc[nearest_indices][targetNull]

      # Find the most frequent targetNull
      most_common_city = Counter(nearest_cities).most_common(1)[0][0]

      # Fill missing targetNull values for the current coordinate
      data.loc[(data[x] == coord[0]) & (data[y] == coord[1]), targetNull] = most_common_city

K_nearthest_places_null_filter('cityname', 'latitude', 'longitude', 1)
K_nearthest_places_null_filter('state', 'latitude', 'longitude', 1)

data['bathrooms'] = data['bathrooms'].fillna(data['bathrooms'].mean())
data['bedrooms']  = data['bedrooms'] .fillna(data['bedrooms'] .mean())

"""convert bathrooms and bedrooms to int because it can't be float values"""

data['bathrooms'] = data['bathrooms'].astype(int)
data['bedrooms'] = data['bedrooms'].astype(int)

print(data['pets_allowed'].value_counts())
data['pets_allowed'].shape

data['cats_allowed'] = 0
data['dogs_allowed'] = 0

data.loc[(data['pets_allowed'] == 'Cats,Dogs') | (data['pets_allowed'] == 'Cats'), 'cats_allowed'] = 1
data.loc[(data['pets_allowed'] == 'Cats,Dogs') | (data['pets_allowed'] == 'Dogs'), 'dogs_allowed'] = 1

print(data['cats_allowed'].value_counts(), data['dogs_allowed'].value_counts())

"""Insert The **cats_allowed** and **dogs_allowed** at it the place of **pets_allowed** then remove **pets_allowed**"""

#          before                                                               after
new_data = pd.concat([data.loc[:, :'pets_allowed'], data[['cats_allowed', 'dogs_allowed']], data.loc[:, 'pets_allowed':].drop(['cats_allowed', 'dogs_allowed'], axis=1)], axis=1)
data = new_data
data.drop(columns='pets_allowed', inplace=True)
print(data['cats_allowed'].shape, data['dogs_allowed'].shape)

"""Remove unwanted column **pets_allowed**"""

print(data.info())

data['address'].fillna('Unknown', inplace=True)
print(data['address'].value_counts())

def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

#remove noise
print(data['price_type'].value_counts())
data = data.drop(data[(data['price_type'] == 'Weekly') | (data['price_type'] == 'Monthly|Weekly')].index)
print(data['price_type'].value_counts())

cols=('title','body','cityname', 'state', 'source', 'address','fee','category','currency','price_type' )
data = Feature_Encoder(data, cols)

data = pd.get_dummies(data, columns = ['has_photo'])

data['has_photo_Yes'] = data['has_photo_Yes'].astype(int)
data['has_photo_Thumbnail'] = data['has_photo_Thumbnail'].astype(int)
data.drop(columns=['has_photo_No'], inplace=True)

def featureScaling(X, a, b):
    X = np.array(X)
    print("Shape of X:", X.shape)  # Add this line to check the shape
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X

data = pd.DataFrame(featureScaling(data, 0, 1), columns=data.columns)

def count_outliers(df):
    for column_name in df.columns:
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
        print(f"Number of outliers in '{column_name}': {outliers.shape[0]}")

outliercolums = ['bathrooms', 'bedrooms', 'square_feet', 'price_display']
print(count_outliers(data[outliercolums]))

sns.boxplot(data['bathrooms'])

sns.boxplot(data['bedrooms'])

sns.boxplot(data['square_feet'])

sns.boxplot(data['price_display'])

def replace_outliers(df, columns):
    for column_name in columns:
        # Calculate quartiles
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        # Calculate IQR
        IQR = Q3 - Q1
        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Replace outliers below the lower bound with Q1 - 1.5 * IQR
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
        # Replace outliers above the upper bound with Q3 + 1.5 * IQR
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound
    return df

data = replace_outliers(data, outliercolums)

print(count_outliers(data[outliercolums]))

print(len(data['title'].value_counts()))
print(len(data['body'].value_counts()))
print(len(data['category'].value_counts()))
print(len(data['currency'].value_counts()))
print(len(data['fee'].value_counts()))
print(len(data['cityname'].value_counts()))
print(len(data['state'].value_counts()))
print(len(data['source'].value_counts()))
print(len(data['address'].value_counts()))
print(len(data['price_type'].value_counts()))
#all value count = 0 , NaN after normalization should be dropped

data = data.drop('category', axis=1)
data = data.drop('currency', axis=1)
data = data.drop('fee', axis=1)
data = data.drop('price_type', axis=1)

c = data.corr()
top_features = c.index[abs(c['price_display'])>=0.15]
plt.subplots(figsize=(20, 15))
top_corr = data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
print(top_features)
top_features = top_features.delete(-1)
print(top_features)

X = data[top_features]
Y = data['price_display']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

poly_features = PolynomialFeatures(degree = 4)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y_train)
y_train_predicted = poly_model.predict(X_train_poly)
ypred=poly_model.predict(poly_features.transform(X_test))

prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('training Mean Square Error', metrics.mean_squared_error(Y_train, y_train_predicted))
print('Mean Square Error', metrics.mean_squared_error(Y_test, prediction))
print('r2_score', r2_score(Y_test, prediction))

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
r2 = model.score(X_test, Y_test)
print("R^2 score:", r2)

model = RandomForestRegressor(n_estimators=40, random_state=42)  # For regression
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
r2 = r2_score(Y_test, y_pred)
mse = metrics.mean_squared_error(Y_test, y_pred)
print("R^2 score:", r2)
print("Mean Squared Error:", mse)

