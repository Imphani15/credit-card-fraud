import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from collections import Counter

df = pd.read_csv('Deep Sight Analytics creditcard_cc.csv')
df.isnull().sum()


df.isnull().shape[0]
print("Non-missing values: " + str(df.isnull().shape[0]))
print("Missing values: " + str(df.shape[0] - df.isnull().shape[0]))

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler().fit(df[["Time", "Amount"]])
df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

pd.concat([df.head(), df.tail()])

print(df.shape)

y = df["Class"] # target 
X = df.iloc[:,0:30]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from imblearn.over_sampling import SMOTE

X_resampled, Y_resampled = SMOTE().fit_resample(X, y)
print("Resampled shape of X: ", X_resampled.shape)
print("Resampled shape of Y: ", Y_resampled.shape)

value_counts = Counter(Y_resampled)
print(value_counts)

X_r_train, X_r_test, y_r_train, y_r_test  = train_test_split(X_resampled, Y_resampled, test_size= 0.2, random_state= 42)

params = {'learning_rate' : 0.2, 
          'max_depth' : 2, 
          'n_estimators' : 200, 
          'subsample': 0.9,
          'objective': 'binary:logistic'}
xgb_model = XGBClassifier(params = params)
xgb_model.fit(X_r_train, y_r_train)

y_pred = xgb_model.predict(X_r_test)

accuracy = np.mean(y_pred == y_r_test)
precision = np.sum((y_pred == 1) & (y_r_test == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_pred == 1) & (y_r_test == 1)) / np.sum(y_r_test == 1)
f1_score = 2 * precision * recall / (precision + recall)
# Print evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1_score)

y_proba = xgb_model.predict_proba(X_r_test)
fraud_probabilities = y_proba[:, 1] 

y_proba = y_proba * (100 - 0) + 0

print(y_proba)# get the probability of the positive class (fraudulent transactions)


# # Save the model as a pickle file
# with open('xgb_model.pkl', 'wb') as file:
#     pickle.dump(xgb_model, file)
