# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 18:27:30 2021

@author: Harish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

train_df=pd.read_excel(r"C:\Users\Harish\Documents\Projects\Flight Price Prediction\Data_Train.xlsx")
test_df=pd.read_excel(r"C:\Users\Harish\Documents\Projects\Flight Price Prediction\Test_set.xlsx")

train_df.shape
test_df.shape

df = train_df.append(test_df,sort=False)
df.dtypes

# Feature Engineering

df['Date'] = df['Date_of_Journey'].str.split('/').str[0]
df['Month'] = df['Date_of_Journey'].str.split('/').str[1]
df['Year'] = df['Date_of_Journey'].str.split('/').str[2]


df['Date'] = df['Date'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Year'] = df['Year'].astype(int)

df = df.drop(['Date_of_Journey'],axis=1)
df.dtypes

df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]

df[df['Total_Stops'].isnull()]

df['Total_Stops'] = df['Total_Stops'].fillna('1 stop')

df['Total_Stops'] = df['Total_Stops'].replace('non-stop','0 stop')

df['Stop'] = df['Total_Stops'].str.split(' ').str[0]

df.head()
df.dtypes

df['Stop'] = df['Stop'].astype(int)
df = df.drop(['Total_Stops'],axis=1)

df['Arrival_Hour'] = df['Arrival_Time'] .str.split(':').str[0]
df['Arrival_Minute'] = df['Arrival_Time'] .str.split(':').str[1]


df['Arrival_Hour'] = df['Arrival_Hour'].astype(int)
df['Arrival_Minute'] = df['Arrival_Minute'].astype(int)
df = df.drop(['Arrival_Time'],axis=1)


df['Departure_Hour'] = df['Dep_Time'] .str.split(':').str[0]
df['Departure_Minute'] = df['Dep_Time'] .str.split(':').str[1]

df['Departure_Hour'] = df['Departure_Hour'].astype(int)
df['Departure_Minute'] = df['Departure_Minute'].astype(int)
df = df.drop(['Dep_Time'],axis=1)


df['Route_1'] = df['Route'].str.split('→ ').str[0]
df['Route_2'] = df['Route'].str.split('→ ').str[1]
df['Route_3'] = df['Route'].str.split('→ ').str[2]
df['Route_4'] = df['Route'].str.split('→ ').str[3]
df['Route_5'] = df['Route'].str.split('→ ').str[4]


df['Price'].fillna((df['Price'].mean()),inplace=True)

df['Route_1'].fillna("None",inplace=True)
df['Route_2'].fillna("None",inplace=True)
df['Route_3'].fillna("None",inplace=True)
df['Route_4'].fillna("None",inplace=True)
df['Route_5'].fillna("None",inplace=True)


df = df.drop(['Route'],axis=1)
df = df.drop(['Duration'],axis=1)

encoder = LabelEncoder()

df["Airline"] = encoder.fit_transform(df['Airline'])
df["Source"] = encoder.fit_transform(df['Source'])
df["Destination"] = encoder.fit_transform(df['Destination'])
df["Additional_Info"] = encoder.fit_transform(df['Additional_Info'])
df["Route_1"] = encoder.fit_transform(df['Route_1'])
df["Route_2"] = encoder.fit_transform(df['Route_2'])
df["Route_3"] = encoder.fit_transform(df['Route_3'])
df["Route_4"] = encoder.fit_transform(df['Route_4'])
df["Route_5"] = encoder.fit_transform(df['Route_5'])


df.isnull().sum()

df_train = df[0:10683]
df_test = df[10683:]

X = df_train.drop(['Price'],axis=1)
y = df_train.Price

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

model = SelectFromModel(Lasso(alpha=0.005,random_state=0))

model.fit(X_train,y_train)

model_l = Lasso(alpha=0.005,random_state=0)

model_l.fit(X_train,y_train)

y_pred_l = model_l.predict(X_test)

score_l = mean_squared_error(y_test, y_pred_l)
scormae_l = mean_absolute_error(y_test, y_pred_l)

score_l
scormae_l

model.get_support()
selected_features = X_train.columns[(model.get_support())]
selected_features

X_train = X_train.drop(['Year'],axis=1)

X_test = X_test.drop(['Year'],axis=1)

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = 1)


rf_random.fit(X_train,y_train)

y_pred = rf_random.predict(X_test)

sns.distplot(y_test-y_pred)

plt.scatter(y_test,y_pred)


score = mean_squared_error(y_test, y_pred)
scormae = mean_absolute_error(y_test, y_pred)
scormae











