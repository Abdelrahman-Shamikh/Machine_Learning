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



data = pd.read_csv('ApartmentRentPrediction_Milestone2.csv')
data.head(10)

"""put the prediction column at the end"""

predicted_col = data['RentCategory']
data.drop(columns=['RentCategory'], inplace=True)
data['RentCategory'] = predicted_col

# Assuming you have a DataFrame named 'df' with your data

# Splitting the DataFrame into features (X) and target variable (y)
X = data.drop(columns=['RentCategory'])
Y = data['RentCategory']

# Splitting the data into training and test sets (80% training, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Concatenating X_train and y_train
train_data = pd.concat([X_train, Y_train], axis=1)

# Concatenating X_test and y_test
test_data = pd.concat([X_test, Y_test], axis=1)

for column in train_data:
    print(train_data[column].value_counts() , '\n')

"""number of nulls in train"""
for column in train_data.columns:
  print(column, '\t', train_data[column].isna().sum())

"""number of nulls in test"""
for column in test_data.columns:
  print(column, '\t', test_data[column].isna().sum())

"""replace each row in amenities with the **count** of the amenities"""
ce = CounterEncoder()
train_data['amenities'] = ce.transform(train_data['amenities'])
test_data['amenities'] = ce.transform(test_data['amenities'])

"""remove noise"""
print(train_data['category'].value_counts())
train_data = train_data.drop(train_data[(train_data['category'] == 'housing/rent/short_term') | (train_data['category'] == 'housing/rent/home')].index)
print(train_data['category'].value_counts())


mam = MeansAndMods()
mam.fit(train_data)

#to be commented
# categoryModeTrain = train_data['category'].mode()[0]

# print(test_data['category'].value_counts())

# for index, value in test_data['category'].items():
#     if value != categoryModeTrain:
#         test_data.at[index, 'category'] = categoryModeTrain

# print(test_data['category'].value_counts())

"""**replace nulls of longitude and latitude first**"""

train_data.dropna(subset=['longitude', 'latitude'], inplace=True)

test_data['longitude'] = mam.transform(test_data['longitude'])
test_data['latitude'] = mam.transform(test_data['latitude'])

# Note: The following line is Required as we want these 2 cols without any nulls


city_KP = KNN_PP()
train_data['cityname'] = city_KP.fit_transform(train_data.loc[:, ['latitude', 'longitude']], train_data['cityname'])
test_data['cityname'] = city_KP.transform(test_data.loc[:, ['latitude', 'longitude']], test_data['cityname'])

state_KP = KNN_PP()
train_data['state'] = state_KP.fit_transform(train_data.loc[:, ['latitude', 'longitude']], train_data['state'])
test_data['state'] = state_KP.transform(test_data.loc[:, ['latitude', 'longitude']], test_data['state'])

train_data['bathrooms'] = mam.transform(train_data['bathrooms'])
train_data['bedrooms']  = mam.transform(train_data['bedrooms'])

train_data['bathrooms'] = train_data['bathrooms'].astype(int)
train_data['bedrooms'] = train_data['bedrooms'].astype(int)

test_data['bathrooms'] = mam.transform(test_data['bathrooms'])
test_data['bedrooms']  = mam.transform(test_data['bedrooms'])

test_data['bathrooms'] = test_data['bathrooms'].astype(int)
test_data['bedrooms'] = test_data['bedrooms'].astype(int)


print(train_data['pets_allowed'].value_counts())

pets_ohe = OHE(removeCol='pets_allowed')
pets_ohe.fit_transform(train_data)
pets_ohe.transform(test_data)

print(train_data['cats_allowed'].value_counts(), train_data['dogs_allowed'].value_counts())

"""number of nulls"""
for column in train_data.columns:
  print(column, '\t',train_data[column].isna().sum())

"""number of nulls"""
for column in test_data.columns:
  print(column, '\t',test_data[column].isna().sum())

train_data['address'].fillna('Unknown', inplace=True)
mam.fit(train_data)
test_data['address'] = mam.transform(test_data['address'])

for col in train_data.columns:
    train_data[col] = mam.transform(train_data[col])
for col in test_data.columns:
    test_data[col] = mam.transform(test_data[col])

"""number of nulls"""
for column in train_data.columns:
  print(column, '\t',train_data[column].isna().sum())

photo_ohe = OHE('has_photo')
photo_ohe.fit_transform(train_data, 'photo')
train_data.drop(columns=['no_photo'], inplace=True)
photo_ohe.transform(test_data)
test_data.drop(columns=['no_photo'], inplace=True)

def count_outliers(df):
    for column_name in df.columns:
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
        print(f"Number of outliers in '{column_name}': {outliers.shape[0]}")

outliercolums = ['bathrooms', 'bedrooms', 'square_feet']
print(count_outliers(train_data[outliercolums]))
# pd.set_option('display.max_rows', None)
print(train_data['bedrooms'].value_counts())

sns.boxplot(train_data['square_feet'])

sns.boxplot(train_data['bedrooms'])

sns.boxplot(train_data['bathrooms'])

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

train_data = replace_outliers(train_data, outliercolums)
print(count_outliers(train_data[outliercolums]))
#print(train_data['square_feet'].max())

train_data.info()

# def Feature_Encoder(X, cols):
#     for c in cols:
#         lbl = LabelEncoder()
#         lbl.fit(list(X[c].values))

#         X[c] = lbl.transform(list(X[c].values))
#     return X

#remove noise
train_data = train_data.drop(train_data[(train_data['price_type'] == 'Weekly') | (train_data['price_type'] == 'Monthly|Weekly')].index)
mam.fit(train_data)

train_data.info()

cols=('category','title','body','cityname', 'state', 'source', 'address','fee','currency','price_type')

lbls = []
for c in cols:
    lbl = LabelEncoder()
    train_data[c] = lbl.fit_transform(list(train_data[c].values))
    lbls.append(lbl)
    encoded_labels = []
    for idx, val in test_data[c].items():
      if val not in lbl.classes_:
        test_data.at[idx,c] = mam.special[c]
    test_data[c] = lbl.transform(test_data[c])

train_data['RentCategory'] = train_data['RentCategory'].replace({'Medium-Priced Rent': 1, 'Low Rent': 0, 'High Rent': 2})
test_data['RentCategory'] = test_data['RentCategory'].replace({'Medium-Priced Rent': 1, 'Low Rent': 0, 'High Rent': 2})

train_data.info()

train_data.head()

from sklearn.preprocessing import MinMaxScaler

RentCategoryData = train_data['RentCategory']
train_data.drop(columns=['RentCategory'], inplace = True)

column_names = train_data.columns

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data.info()
train_data_scaled = scaler.fit_transform(train_data)

# Create a new DataFrame with scaled data
train_data_scaled_df = pd.DataFrame(train_data_scaled, columns=column_names)

train_data = train_data_scaled_df
train_data['RentCategory'] = RentCategoryData.values

print(train_data.head())

print(test_data.head())

RentCategoryDataTest = test_data['RentCategory']
test_data.drop(columns=['RentCategory'], inplace=True)

test_data_scaled = scaler.transform(test_data)

test_data_scaled_df = pd.DataFrame(test_data_scaled, columns=column_names)
test_data= test_data_scaled_df
test_data['RentCategory'] = RentCategoryDataTest.values

print(test_data.head())

train_data.info()

train_data.info()

# train_data.dropna(axis=1, inplace=True)

#train_data.head(7)
train_data.info()

# from sklearn.feature_selection import SelectKBest, chi2

# # Encode categorical variables if not already done. Assuming it's done in your preprocessing.
# # Ensure there are no negative values as chi2 cannot handle them. Chi2 is only for non-negative features and class labels.
# # Since you've handled missing values and encoding, you can directly apply Chi-square.

# # Applying SelectKBest class to extract top 'k' best features
# # Here, 'k' can be any number or you can use 'k='all' to select all features
# k = 10  # Example: select the top 10 features
# bestfeatures = SelectKBest(score_func=chi2, k=k)
# fit = bestfeatures.fit(train_data.drop('RentCategory', axis=1), train_data['RentCategory'])

# # Get the scores for each feature
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(train_data.drop('RentCategory', axis=1).columns)

# # Concatenate two dataframes for better visualization
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Feature', 'Score']  # naming the dataframe columns
# print(featureScores.nlargest(k, 'Score'))  # print k best features

# # You can now decide to keep only the top k features:
# train_data_selected_features = train_data[featureScores.nlargest(k, 'Score')['Feature'].tolist() + ['RentCategory']]

# c = train_data.corr()
# top_features = c.index[abs(c['RentCategory'])>=0.0]
# plt.subplots(figsize=(20, 15))
# top_corr = train_data[top_features].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()
# print(top_features)
# top_features = top_features.delete(-1)
# print(top_features)

categorical_columns = ['category', 'title', 'body', 'currency', 'fee', 'yes_photo', 'thumbnail_photo', 'cats_allowed', 'dogs_allowed', 'price_type', 'address', 'cityname', 'state', 'source']
numerical_columns = ['id', 'bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'time','amenities']

categorical_data = train_data[categorical_columns]
numerical_data = train_data[numerical_columns]

"""## feature Slectoin

### mutual information
"""

def select_best_features_mutual_info(X, Y, num_features):
    # Compute mutual information scores
    mi_scores = mutual_info_classif(X, Y)

    # Create a DataFrame to store feature names and their corresponding mutual information scores
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})

    # Sort features based on mutual information scores (descending order)
    mi_df_sorted = mi_df.sort_values(by='MI_Score', ascending=False)

    # Select top features
    selected_features = mi_df_sorted.iloc[:num_features]['Feature'].tolist()

    # Create DataFrame with selected features
    X_top = X[selected_features]

    return X_top.columns.tolist()

"""### chi-squared"""

def select_best_features_chi2(X, Y, num_features):
    # SelectKBest with chi-squared as the scoring function
    selector = SelectKBest(score_func=chi2, k=num_features)
    X_new = selector.fit_transform(X, Y)

    # Get chi-squared scores and corresponding feature names
    chi2_scores = selector.scores_
    feature_names = X.columns

    # Create a DataFrame to store feature names and their corresponding chi-squared scores
    chi2_df = pd.DataFrame({'Feature': feature_names, 'Chi2_Score': chi2_scores})

    # Sort features based on chi-squared scores (descending order)
    chi2_df_sorted = chi2_df.sort_values(by='Chi2_Score', ascending=False)

    # Select top features
    selected_features = chi2_df_sorted.iloc[:num_features]['Feature'].tolist()

    # Create DataFrame with selected features
    X_top = X[selected_features]

    return X_top.columns.tolist()

"""### anova"""

def select_best_features_anova(X, Y, num_features):

    # Perform ANOVA for feature selection
    best_features = SelectKBest(score_func=f_classif, k=num_features)
    fit = best_features.fit(X, Y)

    # Get ANOVA F-values and corresponding feature names
    f_scores = pd.DataFrame(fit.scores_)
    feature_names = pd.DataFrame(X.columns)

    # Combine feature names and their ANOVA F-values
    feature_scores = pd.concat([feature_names, f_scores], axis=1)
    feature_scores.columns = ['Feature', 'Score']  # Naming the DataFrame columns

    # Sort features based on ANOVA F-values (descending order)
    feature_scores_sorted = feature_scores.sort_values(by='Score', ascending=False)

    # Select top features
    selected_features = feature_scores_sorted.iloc[:num_features]['Feature'].tolist()

    # Create DataFrame with selected features
    X_top = X[selected_features]

    return X_top.columns.tolist()

"""###  kendall"""

def select_best_features_kendall_tau(X, Y, num_features):
    # Compute Kendall's tau between each feature and the target
    scores, pvalues = [], []
    for column in X.columns:
        score, pvalue = kendalltau(X[column], Y)
        scores.append(score if pd.notnull(score) else 0)  # handle NaN scores
        pvalues.append(pvalue)

    # Create DataFrame to view scores and p-values
    dfscores = pd.DataFrame(scores, index=X.columns, columns=['Score'])
    dfpvalues = pd.DataFrame(pvalues, index=X.columns, columns=['PValue'])

    # Combine scores and p-values into a single DataFrame
    feature_scores = pd.concat([dfscores, dfpvalues], axis=1)
    feature_scores.reset_index(inplace=True)
    feature_scores.columns = ['Feature', 'Score', 'PValue']  # Rename the DataFrame columns

    # Select the top k features based on the score
    selected_features = feature_scores.nlargest(num_features, 'Score')['Feature'].tolist()

    # Create DataFrame with selected features
    X_top = X[selected_features]

    return X_top.columns.tolist()

categorical_features = select_best_features_chi2(categorical_data, train_data['RentCategory'], 5)
numerical_features = select_best_features_anova(numerical_data, train_data['RentCategory'], 5)

X_top_features = categorical_features + numerical_features
X_top_features

X_train = train_data[X_top_features]
Y_train = train_data['RentCategory']

X_test = test_data[X_top_features]
Y_test = test_data['RentCategory']

"""## models"""

from sklearn import svm
from sklearn.metrics import accuracy_score
Strain_time = time.time()
svm_classifier = svm.SVC(kernel='linear', C=1)

svm_classifier.fit(X_train, Y_train)

# Make predictions on the training set
Y_train_predicted = svm_classifier.predict(X_train)

# Compute training accuracy
train_accuracy = accuracy_score(Y_train, Y_train_predicted)
print("Train accuracy:", train_accuracy)
Etrain_time = time.time()
print("time of train :",Etrain_time-Strain_time)
Stest_time = time.time()
# Make predictions on the test set
Y_test_predicted = svm_classifier.predict(X_test)

# Compute test accuracy
test_accuracy = accuracy_score(Y_test, Y_test_predicted)
print("Test accuracy:", test_accuracy)
Etest_time = time.time()
print("time of test :",Etest_time-Stest_time)

import matplotlib.pyplot as plt

# Classification accuracy
accuracies = [train_accuracy, test_accuracy]
labels = ['Train Accuracy', 'Test Accuracy']

plt.figure(figsize=(8, 6))
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.title('Classification Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Total training time and total test time
times = [(Etrain_time - Strain_time), (Stest_time - Etrain_time)]
labels = ['Training Time', 'Test Time']

plt.figure(figsize=(8, 6))
plt.bar(labels, times, color=['orange', 'red'])
plt.title('Total Training and Test Time')
plt.ylabel('Time (seconds)')
plt.show()

import time
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier()

# Training
train_start_time = time.time()
xgb_classifier.fit(X_train, Y_train)
train_end_time = time.time()

# Make predictions on the training set
Y_train_predicted = xgb_classifier.predict(X_train)

# Compute training accuracy
train_accuracy = accuracy_score(Y_train, Y_train_predicted)
print("Training accuracy:", train_accuracy)
print("Training time:", train_end_time - train_start_time)

# Testing
test_start_time = time.time()
# Make predictions on the test set
Y_test_predicted = xgb_classifier.predict(X_test)
test_end_time = time.time()

# Compute test accuracy
test_accuracy = accuracy_score(Y_test, Y_test_predicted)
print("Test accuracy:", test_accuracy)
print("Test time:", test_end_time - test_start_time)

# Plotting
labels = ['Train Accuracy', 'Test Accuracy', 'Training Time', 'Test Time']
accuracies = [train_accuracy, test_accuracy]
times = [train_end_time - train_start_time, test_end_time - test_start_time]

plt.figure(figsize=(12, 6))

# Bar graph for classification accuracy
plt.subplot(1, 2, 1)
plt.bar(labels[:2], accuracies, color=['blue', 'green'])
plt.title('Classification Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Bar graph for total training and test time
plt.subplot(1, 2, 2)
plt.bar(labels[2:], times, color=['orange', 'red'])
plt.title('Total Training and Test Time')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()

import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

# Training
train_start_time_rf = time.time()
rf_classifier.fit(X_train, Y_train)
train_end_time_rf = time.time()

# Make predictions on the training set
Y_train_predicted_rf = rf_classifier.predict(X_train)

# Compute training accuracy
train_accuracy_rf = accuracy_score(Y_train, Y_train_predicted_rf)
print("Random Forest Training accuracy:", train_accuracy_rf)
print("Random Forest Training time:", train_end_time_rf - train_start_time_rf)

# Testing
test_start_time_rf = time.time()
# Make predictions on the test set
Y_test_predicted_rf = rf_classifier.predict(X_test)
test_end_time_rf = time.time()

# Compute test accuracy
test_accuracy_rf = accuracy_score(Y_test, Y_test_predicted_rf)
print("Random Forest Test accuracy:", test_accuracy_rf)
print("Random Forest Test time:", test_end_time_rf - test_start_time_rf)

# Plotting
labels_rf = ['Train Accuracy', 'Test Accuracy', 'Training Time', 'Test Time']
accuracies_rf = [train_accuracy_rf, test_accuracy_rf]
times_rf = [train_end_time_rf - train_start_time_rf, test_end_time_rf - test_start_time_rf]

plt.figure(figsize=(12, 6))

# Bar graph for classification accuracy
plt.subplot(1, 2, 1)
plt.bar(labels_rf[:2], accuracies_rf, color=['blue', 'green'])
plt.title('Random Forest Classification Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Bar graph for total training and test time
plt.subplot(1, 2, 2)
plt.bar(labels_rf[2:], times_rf, color=['orange', 'red'])
plt.title('Random Forest Total Training and Test Time')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()

import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create KNN model
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Training
train_start_time_knn = time.time()
knn_classifier.fit(X_train, Y_train)
train_end_time_knn = time.time()

# Make predictions on the training set
Y_train_predicted_knn = knn_classifier.predict(X_train)

# Compute training accuracy
train_accuracy_knn = accuracy_score(Y_train, Y_train_predicted_knn)
print("KNN Training accuracy:", train_accuracy_knn)
print("KNN Training time:", train_end_time_knn - train_start_time_knn)

# Testing
test_start_time_knn = time.time()
# Make predictions on the test set
Y_test_predicted_knn = knn_classifier.predict(X_test)
test_end_time_knn = time.time()

# Compute test accuracy
test_accuracy_knn = accuracy_score(Y_test, Y_test_predicted_knn)
print("KNN Test accuracy:", test_accuracy_knn)
print("KNN Test time:", test_end_time_knn - test_start_time_knn)

# Plotting
labels_knn = ['Train Accuracy', 'Test Accuracy', 'Training Time', 'Test Time']
accuracies_knn = [train_accuracy_knn, test_accuracy_knn]
times_knn = [train_end_time_knn - train_start_time_knn, test_end_time_knn - test_start_time_knn]

plt.figure(figsize=(12, 6))

# Bar graph for classification accuracy
plt.subplot(1, 2, 1)
plt.bar(labels_knn[:2], accuracies_knn, color=['blue', 'green'])
plt.title('KNN Classification Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Bar graph for total training and test time
plt.subplot(1, 2, 2)
plt.bar(labels_knn[2:], times_knn, color=['orange', 'red'])
plt.title('KNN Total Training and Test Time')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()

import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create logistic regression model
logreg = LogisticRegression()

# Training
train_start_time_lr = time.time()
logreg.fit(X_train, Y_train)
train_end_time_lr = time.time()

# Make predictions on the training set
Y_train_predicted_lr = logreg.predict(X_train)

# Compute training accuracy
train_accuracy_lr = accuracy_score(Y_train, Y_train_predicted_lr)
print("Logistic Regression Training accuracy:", train_accuracy_lr)
print("Logistic Regression Training time:", train_end_time_lr - train_start_time_lr)

# Testing
test_start_time_lr = time.time()
# Make predictions on the test set
Y_test_predicted_lr = logreg.predict(X_test)
test_end_time_lr = time.time()

# Compute test accuracy
test_accuracy_lr = accuracy_score(Y_test, Y_test_predicted_lr)
print("Logistic Regression Test accuracy:", test_accuracy_lr)
print("Logistic Regression Test time:", test_end_time_lr - test_start_time_lr)

# Plotting
labels_lr = ['Train Accuracy', 'Test Accuracy', 'Training Time', 'Test Time']
accuracies_lr = [train_accuracy_lr, test_accuracy_lr]
times_lr = [train_end_time_lr - train_start_time_lr, test_end_time_lr - test_start_time_lr]

plt.figure(figsize=(12, 6))

# Bar graph for classification accuracy
plt.subplot(1, 2, 1)
plt.bar(labels_lr[:2], accuracies_lr, color=['blue', 'green'])
plt.title('Logistic Regression Classification Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Bar graph for total training and test time
plt.subplot(1, 2, 2)
plt.bar(labels_lr[2:], times_lr, color=['orange', 'red'])
plt.title('Logistic Regression Total Training and Test Time')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()

import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Initialize classifiers
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = SVC(probability=True)

# Create Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3)], voting='soft')

# Training
train_start_time_voting = time.time()
voting_clf.fit(X_train, Y_train)
train_end_time_voting = time.time()

# Make predictions on the training set
y_pred_train_voting = voting_clf.predict(X_train)

# Compute training accuracy
train_accuracy_voting = accuracy_score(Y_train, y_pred_train_voting)
print("Voting Classifier Training accuracy:", train_accuracy_voting)
print("Voting Classifier Training time:", train_end_time_voting - train_start_time_voting)

# Testing
test_start_time_voting = time.time()
# Make predictions on the test set
y_pred_test_voting = voting_clf.predict(X_test)
test_end_time_voting = time.time()

# Compute test accuracy
test_accuracy_voting = accuracy_score(Y_test, y_pred_test_voting)
print("Voting Classifier Test accuracy:", test_accuracy_voting)
print("Voting Classifier Test time:", test_end_time_voting - test_start_time_voting)

# Plotting
labels_voting = ['Train Accuracy', 'Test Accuracy', 'Training Time', 'Test Time']
accuracies_voting = [train_accuracy_voting, test_accuracy_voting]
times_voting = [train_end_time_voting - train_start_time_voting, test_end_time_voting - test_start_time_voting]

plt.figure(figsize=(12, 6))

# Bar graph for classification accuracy
plt.subplot(1, 2, 1)
plt.bar(labels_voting[:2], accuracies_voting, color=['blue', 'green'])
plt.title('Voting Classifier Classification Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Bar graph for total training and test time
plt.subplot(1, 2, 2)
plt.bar(labels_voting[2:], times_voting, color=['orange', 'red'])
plt.title('Voting Classifier Total Training and Test Time')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()

import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# Initialize classifiers
clf1 = DecisionTreeClassifier()
clf2 = SVC(probability=True)
clf3 = RandomForestClassifier()
meta_clf = LogisticRegression()

# Create Stacking Classifier
stacking_clf = StackingClassifier(estimators=[('dt', clf1), ('svm', clf2), ('rf', clf3)], final_estimator=meta_clf)

# Training
train_start_time_stacking = time.time()
stacking_clf.fit(X_train, Y_train)
train_end_time_stacking = time.time()

# Make predictions on the training set
y_pred_train_stacking = stacking_clf.predict(X_train)

# Compute training accuracy
train_accuracy_stacking = accuracy_score(Y_train, y_pred_train_stacking)
print("Stacking Classifier Training accuracy:", train_accuracy_stacking)
print("Stacking Classifier Training time:", train_end_time_stacking - train_start_time_stacking)

# Testing
test_start_time_stacking = time.time()
# Make predictions on the test set
y_pred_test_stacking = stacking_clf.predict(X_test)
test_end_time_stacking = time.time()

# Compute test accuracy
test_accuracy_stacking = accuracy_score(Y_test, y_pred_test_stacking)
print("Stacking Classifier Test accuracy:", test_accuracy_stacking)
print("Stacking Classifier Test time:", test_end_time_stacking - test_start_time_stacking)

# Plotting
labels_stacking = ['Train Accuracy', 'Test Accuracy', 'Training Time', 'Test Time']
accuracies_stacking = [train_accuracy_stacking, test_accuracy_stacking]
times_stacking = [train_end_time_stacking - train_start_time_stacking, test_end_time_stacking - test_start_time_stacking]

plt.figure(figsize=(12, 6))

# Bar graph for classification accuracy
plt.subplot(1, 2, 1)
plt.bar(labels_stacking[:2], accuracies_stacking, color=['blue', 'green'])
plt.title('Stacking Classifier Classification Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Bar graph for total training and test time
plt.subplot(1, 2, 2)
plt.bar(labels_stacking[2:], times_stacking, color=['orange', 'red'])
plt.title('Stacking Classifier Total Training and Test Time')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()

with open('testScript.pkl', 'wb') as f:
    pickle.dump(rf_classifier,f)
    pickle.dump(ce, f)
    pickle.dump(mam, f)
    pickle.dump(city_KP, f)
    pickle.dump(state_KP, f)
    pickle.dump(photo_ohe, f)
    pickle.dump(pets_ohe, f)
    pickle.dump(scaler, f)
    pickle.dump(lbls, f)
