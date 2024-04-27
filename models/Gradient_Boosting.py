##########################
### IMPORT LIBRARIES
##########################

import time
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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
print('Begin training Gradient Boosting Decision Trees...')
gradientboost = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, random_state = 0).fit(X_train,y_train)

gradient_accuracy = accuracy_score(y_true = y_valid, y_pred = gradientboost.predict(X_valid))
gradient_precision = precision_score(y_true = y_valid, y_pred = gradientboost.predict(X_valid))
gradient_recall = recall_score(y_true = y_valid, y_pred = gradientboost.predict(X_valid))
gradient_auc = roc_auc_score(y_true = y_valid, y_score = gradientboost.predict_proba(X_valid)[:, 1])

print(f"Decision Trees Training Accuracy: {round(gradientboost.score(X_train, y_train), 4)}")
print('\n')
print(f"Decision Trees Accuracy: {round(gradient_accuracy, 4)}")
print(f"Decision Trees Precision: {round(gradient_precision, 4)}")
print(f"Decision Trees Recall: {round(gradient_recall, 4)}")
print(f"Decision Trees AUC: {round(gradient_auc, 4)}")
print('\n')
print(f"Decision Trees Test Accuracy: {round(gradientboost.score(X_test, y_test), 4)}")

print('End training Gradient Boosting Decision Trees...')
end_time = time.time()
print('\n')
print(f"Training Time: {round(end_time - start_time, 2)} seconds")

##########################
### RESULTS
##########################

# Decision Trees Training Accuracy: 0.7772

# Decision Trees Accuracy: 0.765
# Decision Trees Precision: 0.7086
# Decision Trees Recall: 0.6765
# Decision Trees AUC: 0.834

# Decision Trees Test Accuracy: 0.7702

# Training Time: 372.74 seconds