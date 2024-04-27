
##########################
### IMPORT LIBRARIES
##########################

import time
import pickle
import numpy as np
from sklearn import linear_model
from sklearn.metrics import  (
                                precision_score, 
                                recall_score,accuracy_score, 
                                roc_auc_score
                                )

##########################
### UNPICKLING DATA
##########################

print('Begin unpickling data...')
for name in ['X_train', 'X_test', 'X_valid', 'y_train', 'y_test', 'y_valid']:
    print(f'Unpickling {name}...')
    with open(f'pickle_files\\{name}.pkl', 'rb') as f:
        globals()[name] = pickle.load(f)
print('End unpickling data...')
print('\n')

##########################
### CONVERTING DATA
##########################

print('Begin converting X tensors to numpy arrays...')
for name in ['X_train', 'X_valid', 'X_test']:
    print(f'Converting {name} to numpy arrays...')
    globals()[name] = [tensor.numpy().flatten() for tensor in globals()[name]]
print('Begin converting X tensors to numpy arrays...')
print('\n')

print('Begin converting y tensors to numpy arrays...')
for name in ['y_train', 'y_valid', 'y_test']:
    print(f'Converting {name} to numpy arrays...')
    globals()[name] = globals()[name].numpy()
print('End converting y tensors to numpy arrays...')
print('\n')

##########################
### TRAINING
##########################

start_time = time.time()
print('Begin training Logistic Regression...')
logistic = linear_model.LogisticRegression(max_iter = 5000, C = 370).fit(X_train, y_train)

logistic_accuracy = accuracy_score(y_true = y_valid, y_pred = logistic.predict(X_valid))
logistic_precision = precision_score(y_true = y_valid, y_pred = logistic.predict(X_valid))
logistic_recall = recall_score(y_true = y_valid, y_pred = logistic.predict(X_valid))
logistic_auc = roc_auc_score(y_true = y_valid, y_score = logistic.predict_proba(X_valid)[:, 1])

print(f"Logistic Regression Training Accuracy: {round(logistic.score(X_train, y_train), 4)}")
print(f"Logistic Regression Accuracy: {round(logistic_accuracy, 4)}")
print(f"Logistic Regression Precision: {round(logistic_precision, 4)}")
print(f"Logistic Regression Recall: {round(logistic_recall, 4)}")
print(f"Logistic Regression AUC: {round(logistic_auc, 4)}")
print('\n')
print(f"Logistic Regression Test Accuracy: {round(logistic.score(X_test, y_test), 4)}")

print('End training Logistic Regression...')
end_time = time.time()
print('\n')
print(f"Training Time: {round(end_time - start_time, 2)} seconds")

##########################
### RESULTS
##########################

# Logistic Regression Training Accuracy: 0.74
# Logistic Regression Accuracy: 0.7372
# Logistic Regression Precision: 0.7247
# Logistic Regression Recall: 0.7651
# Logistic Regression AUC: 0.8009

# Logistic Regression Test Accuracy: 0.7347

# Training Time: 20.53 seconds
