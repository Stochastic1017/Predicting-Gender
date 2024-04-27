
##########################
### IMPORT LIBRARIES
##########################

import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, roc_auc_score

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### UNPICKLING DATA
##########################

for name in ['X_train', 'X_test', 'X_valid', 'y_train', 'y_test', 'y_valid']:
    with open(f'pickle_files\\{name}.pkl', 'rb') as f:
        globals()[name] = pickle.load(f)

##########################
### DATA PREPARATION
##########################

class CustomNameDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].reshape(1, 25, 81), self.y[idx]
    
train_dataset = CustomNameDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = 64)

validation_dataset = CustomNameDataset(X_valid, y_valid)
validation_loader = DataLoader(validation_dataset, shuffle = False, batch_size = 64)

test_dataset = CustomNameDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = 64)

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.05
num_epochs = 10
batch_size = 64

# Architecture
num_features = 25 * 81
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 2

##########################
### MODEL
##########################

class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 8, kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.conv_2 = torch.nn.Conv2d(8, 16, kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.pool_2 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.linear_1 = torch.nn.Linear(16 * 6 * 20, num_classes)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
        x = x.view(-1, 16 * 6 * 20)  # Flatten the tensor for the linear layer
        logits = self.linear_1(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

torch.manual_seed(random_seed)
model = ConvNet(num_classes = num_classes)

model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 

def compute_metrics_and_accuracy(model, data_loader):
    model.eval()
    correct_preds, total_samples = 0, 0
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.view(-1, 1, 25, 81).to(device), targets.to(device)
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            correct_preds += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())
    accuracy = correct_preds / total_samples * 100
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_predictions)
    return accuracy, precision, recall, auc

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.view(-1, 1, 25, 81).to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets.long())
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                    %(epoch+1, num_epochs, batch_idx, 
                    len(train_loader), cost))
            
    train_accuracy, _, _, _ = compute_metrics_and_accuracy(model, train_loader)
    validation_accuracy, precision, recall, auc = compute_metrics_and_accuracy(model, validation_loader)
    print('\n')
    print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | Training Accuracy: {train_accuracy:.2f}% | Validation Accuracy: {validation_accuracy:.2f}%')
    print(f'Validation Precision: {precision:.2f}, Validation Recall: {recall:.2f}, Validation AUC: {auc:.2f}\n')

total_training_time = (time.time() - start_time) / 60
print(f'Total Training Time: {total_training_time:.2f} min')

test_accuracy = compute_metrics_and_accuracy(model, test_loader)[0]
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Training Accuracy: 76.20% 

# Validation Accuracy: 75.56%
# Validation Precision: 0.70
# Validation Recall: 0.65
# Validation AUC: 0.74

# Test Accuracy: 76.58%

# Total Training Time: 2.99 min