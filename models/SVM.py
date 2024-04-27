
##########################
### IMPORT LIBRARIES
##########################

import pickle
import numpy
from sklearn import svm
from sklearn.metrics import (
                             precision_score, 
                             recall_score,
                             accuracy_score, 
                             roc_auc_score, 
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

print('Begin training SVM model...')
clf = svm.SVC(kernel = 'rbf').fit(X_train, y_train)
print('End training SVM model...')

svm_accuracy = accuracy_score(y_valid, clf.predict(X_valid))
svm_precision = precision_score(y_valid, clf.predict(X_valid))
svm_recall = recall_score(y_valid, clf.predict(X_valid))
svm_auc = roc_auc_score(y_valid, clf.predict(X_valid)[:, 1])

print(f"SVM Training Accuracy: {round(clf.score(X_train, y_train), 4)}")
print(f"SVM Validation Accuracy: {round(svm_accuracy, 4)}")
print(f"SVM Validation Precision: {round(svm_precision, 4)}")
print(f"SVM Validation Recall: {round(svm_recall, 4)}")
print(f"SVM Validation AUC: {round(svm_auc, 4)}")
print('\n')
print(f"SVM Test Accuracy: {round(clf.score(X_test, y_test), 4)}")

#K-nearest Neighbors Accuracy: 0.7091
#K-nearest Neighbors Precision: 0.6432
#K-nearest Neighbors Recall: 0.573
#K-nearest Neighbors AUC: 0.7459