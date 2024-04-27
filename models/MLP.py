
##########################
### IMPORT LIBRARIES
##########################

import time
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, roc_auc_score

##########################
### UNPICKLING DATA
##########################

print('Begin unpickling data...')
for name in ['X_train', 'X_test', 'X_valid', 'y_train', 'y_test', 'y_valid']:
    print(f'Unpickling {name}...')
    with open(f'C:\\Users\\shriv\\OneDrive\\Documents\\names_gender_project\\data_pickle\\{name}.pkl', 'rb') as f:
        globals()[name] = pickle.load(f)
print('End unpickling data...')

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.05
num_epochs = 15
batch_size = 256
dropout_prob = 0.5

# Architecture
num_features = 2025
num_hidden_1 = 1024
num_hidden_2 = 512
num_hidden_3 = 256
num_classes = 2

##########################
### DATA PREPARATION
##########################

print('Preparing data...')
class CustomNameDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
print('Preparing train loader...')
train_dataset = CustomNameDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)

print('Preparing validation loader...')
validation_dataset = CustomNameDataset(X_valid, y_valid)
validation_loader = DataLoader(validation_dataset, shuffle = True, batch_size = batch_size)

print('Preparing test loader...')
test_dataset = CustomNameDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, shuffle = True, batch_size = batch_size)

##########################
### MODEL ARCHITECTURE
##########################

class MultilayerPerceptron(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(MultilayerPerceptron, self).__init__()
        
        ### 1st hidden layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.linear_1_bn = torch.nn.BatchNorm1d(num_hidden_1)
        
        ### 2nd hidden layer
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_2_bn = torch.nn.BatchNorm1d(num_hidden_2)

        ### 3rd hidden layer
        self.linear_3 = torch.nn.Linear(num_hidden_2, num_hidden_3)
        self.linear_3_bn = torch.nn.BatchNorm1d(num_hidden_3)

        ### Output layer
        self.linear_out = torch.nn.Linear(num_hidden_3, num_classes)
        
    def forward(self, x):
        out = self.linear_1(x)
        out = self.linear_1_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p = dropout_prob, training = self.training)

        out = self.linear_2(out)
        out = self.linear_2_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p = dropout_prob, training = self.training)

        out = self.linear_3(out)
        out = self.linear_3_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p = dropout_prob, training = self.training)

        logits = self.linear_out(out)
        probas = F.log_softmax(logits, dim = 1)
        return logits, probas

##########################
### TRAINING LOOP
##########################

torch.manual_seed(random_seed)
model = MultilayerPerceptron(   num_features = num_features,
                                num_classes = num_classes    )

model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)  

def compute_metrics_and_accuracy(model, data_loader):
    model.eval()
    correct_preds, total_samples = 0, 0
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.view(-1, num_features).to(device), targets.to(device)
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            correct_preds += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())
    accuracy = correct_preds / total_samples
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_predictions)
    return accuracy, precision, recall, auc

start_time = time.time()
validation_accuracies = []
validation_precision = []
validation_recall = []
validation_auc = []
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.view(-1, num_features).to(device)
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
    print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | Training Accuracy: {train_accuracy:.2f} | Validation Accuracy: {validation_accuracy:.2f}')
    print(f'Validation Precision: {precision:.2f}, Validation Recall: {recall:.2f}, Validation AUC: {auc:.2f}\n')

    validation_accuracies.append(validation_accuracy)
    validation_precision.append(precision)
    validation_recall.append(recall)
    validation_auc.append(auc)

total_training_time = (time.time() - start_time) / 60
print(f'Total Training Time: {total_training_time:.2f} min')

test_accuracy = compute_metrics_and_accuracy(model, test_loader)[0]
print(f'Test Accuracy: {test_accuracy:.2f}%')

plt.figure(figsize = (10, 6))

plt.plot(validation_precision, color = 'blue', label = 'Validation Precision')
plt.plot(validation_recall, color = 'green', label = 'Validation Recall')
plt.plot(validation_auc, color = 'yellow', label = 'Validation AUC')
plt.plot(validation_accuracies, color = 'red', label = 'Validation accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()

# Training Accuracy: 82.33% 
# Validation Accuracy: 79.55%
# Validation Precision: 0.78
# Validation Recall: 0.83
# Validation AUC: 0.80

# Total Training Time: 93 seconds
# Test Accuracy: 79.03%