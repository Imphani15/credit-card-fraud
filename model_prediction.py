# import libraries
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np


with open('Models/xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

df = pd.read_csv('CSV_Folder/shortdata1.csv')
df.isnull().sum()
df.isnull().shape[0]

# print("Non-missing values: " + str(df.isnull().shape[0]))
# print("Missing values: " + str(df.shape[0] - df.isnull().shape[0]))

scaler = RobustScaler().fit(df[["Time", "Amount"]])
df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

predictions = xgb_model.predict(df)

y_proba = xgb_model.predict_proba(df)*100
y_proba = np.round(y_proba, decimals=0)

fraud_probabilities = y_proba[:, 1] 

print(y_proba)
