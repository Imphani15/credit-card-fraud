import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv('creditcard.csv')
df.isnull().sum()
df['Class'].value_counts()
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

df.drop('Time', axis = 1, inplace = True)
#Splitting the data into feature & targets
X = df.drop(columns = 'Class', axis = 1) #1 = column, 0 = row 
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10)
oversampler = RandomOverSampler(random_state=5)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)



scaler = StandardScaler()
X_train_resampled['Amount'] = scaler.fit_transform(X_train_resampled[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])



params = {'learning_rate' : 0.2, 
          'max_depth' : 2, 
          'n_estimators' : 200, 
          'subsample': 0.9,
          'objective': 'binary:logistic'}
xgb_model = XGBClassifier(params = params)
xgb_model.fit(X_train_resampled, y_train_resampled)

y_pred = xgb_model.predict(X_test)


accuracy = np.mean(y_pred == y_test)
precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
f1_score = 2 * precision * recall / (precision + recall)
auc_score = xgb_model.predict_proba(X_test)[:, 1]

# Print evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1_score)


# Save the model as a pickle file
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)