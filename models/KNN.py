
##########################
### IMPORT LIBRARIES
##########################

import time
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
print('Begin training K-nearest Neighbors...')
knn = KNeighborsClassifier(metric = 'cosine', n_neighbors = 3).fit(X_train, y_train)

knn_accuracy = accuracy_score(y_true = y_valid, y_pred = knn.predict(X_valid))
knn_precision = precision_score(y_true = y_valid, y_pred = knn.predict(X_valid))
knn_recall = recall_score(y_true = y_valid, y_pred = knn.predict(X_valid))
knn_auc = roc_auc_score(y_true = y_valid, y_score = knn.predict_proba(X_valid)[:, 1])

print(f"K-Nearest Neighbors Training Accuracy: {round(knn.score(X_train, y_train), 4)}")
print(f"K-nearest Neighbors Accuracy: {round(knn_accuracy, 4)}")
print(f"K-nearest Neighbors Precision: {round(knn_precision, 4)}")
print(f"K-nearest Neighbors Recall: {round(knn_recall, 4)}")
print(f"K-nearest Neighbors AUC: {round(knn_auc, 4)}")
print('\n')
print(f"K-Nearest Neighbors Test Accuracy: {round(knn.score(X_test, y_test), 4)}")

print('End training K-nearest Neighbors...')
end_time = time.time()
print('\n')
print(f"Training Time: {round(end_time - start_time, 2)} seconds")

##########################
### RESULTS
##########################

# K-Nearest Neighbors Training Accuracy: 0.8534

# K-nearest Neighbors Accuracy: 0.7071
# K-nearest Neighbors Precision: 0.64
# K-nearest Neighbors Recall: 0.5713
# K-nearest Neighbors AUC: 0.7437

# K-Nearest Neighbors Test Accuracy: 0.7091

# Training Time: 365.08 seconds