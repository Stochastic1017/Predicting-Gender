##########################
### IMPORT LIBRARIES
##########################

import time
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
print('Begin training Decision Trees...')
decision_tree = DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)

tree_accuracy = accuracy_score(y_true = y_valid, y_pred = decision_tree.predict(X_valid))
tree_precision = precision_score(y_true = y_valid, y_pred = decision_tree.predict(X_valid))
tree_recall = recall_score(y_true = y_valid, y_pred = decision_tree.predict(X_valid))
tree_auc = roc_auc_score(y_true = y_valid, y_score = decision_tree.predict_proba(X_valid)[:, 1])

print(f"Decision Trees Training Accuracy: {round(decision_tree.score(X_train, y_train), 4)}")
print('\n')
print(f"Decision Trees Accuracy: {round(tree_accuracy, 4)}")
print(f"Decision Trees Precision: {round(tree_precision, 4)}")
print(f"Decision Trees Recall: {round(tree_recall, 4)}")
print(f"Decision Trees AUC: {round(tree_auc, 4)}")
print('\n')
print(f"Decision Trees Test Accuracy: {round(decision_tree.score(X_test, y_test), 4)}")

print('End training Decision Trees...')
end_time = time.time()
print('\n')
print(f"Training Time: {round(end_time - start_time, 2)} seconds")

##########################
### RESULTS
##########################

# Decision Trees Training Accuracy: 0.9273

# Decision Trees Accuracy: 0.6785
# Decision Trees Precision: 0.5988
# Decision Trees Recall: 0.5356
# Decision Trees AUC: 0.6728

# Decision Trees Test Accuracy: 0.6786

# Training Time: 13.99 seconds