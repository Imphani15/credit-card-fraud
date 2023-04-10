import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('creditcard.csv')
df.isnull().sum()


df.isnull().shape[0]
print("Non-missing values: " + str(df.isnull().shape[0]))
print("Missing values: " + str(df.shape[0] - df.isnull().shape[0]))

scaler = RobustScaler().fit(df[["Time", "Amount"]])
df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

pd.concat([df.head(), df.tail()])

print(df.shape)

y = df["Class"] # target 
X = df.iloc[:,0:30]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


X_resampled, Y_resampled = SMOTE().fit_resample(X, y)
print("Resampled shape of X: ", X_resampled.shape)
print("Resampled shape of Y: ", Y_resampled.shape)

value_counts = Counter(Y_resampled)
print(value_counts)

X_r_train, X_r_test, y_r_train, y_r_test  = train_test_split(X_resampled, Y_resampled, test_size= 0.2, random_state= 42)

logistic_model = LogisticRegression(C = 0.01)
logistic_model.fit(X_r_train, y_r_train)

y_pred = logistic_model.predict(X_r_test)

accuracy = np.mean(y_pred == y_r_test)
precision = np.sum((y_pred == 1) & (y_r_test == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_pred == 1) & (y_r_test == 1)) / np.sum(y_r_test == 1)
f1_score = 2 * precision * recall / (precision + recall)
# Print evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1_score)


'''# df['Class'].value_counts()
# fraud = df[df['Class'] == 1]
# legit = df[df['Class'] == 0]

# df.drop('Time', axis = 1, inplace = True)
# #Splitting the data into feature & targets
# X = df.drop(columns = 'Class', axis = 1) #1 = column, 0 = row 
# y = df['Class']
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10)
# oversampler = RandomOverSampler(random_state=5)
# X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)



# scaler = StandardScaler()
# X_train_resampled['Amount'] = scaler.fit_transform(X_train_resampled[['Amount']])
# X_test['Amount'] = scaler.transform(X_test[['Amount']])



# params = {'learning_rate' : 0.2, 
#           'max_depth' : 2, 
#           'n_estimators' : 200, 
#           'subsample': 0.9,
#           'objective': 'binary:logistic'}
# xgb_model = XGBClassifier(params = params)
# xgb_model.fit(X_train_resampled, y_train_resampled)

# y_pred = xgb_model.predict(X_test)


# accuracy = np.mean(y_pred == y_test)
# precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
# recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
# f1_score = 2 * precision * recall / (precision + recall)
# auc_score = xgb_model.predict_proba(X_test)[:, 1]

# # Print evaluation metrics
# print('Accuracy:', accuracy)
# print('Precision:', precision)
# print('Recall:', recall)
# print('F1-Score:', f1_score)


# # Save the model as a pickle file
# with open('xgb_model.pkl', 'wb') as file:
#     pickle.dump(xgb_model, file)
# '''