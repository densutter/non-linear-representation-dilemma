import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import copy

class MLPBlock(nn.Module):
    def __init__(self, in_features=16, out_features=16, dropout_prob=0.0):
        super(MLPBlock, self).__init__()
        self.ff1 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.ff1(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class MLPModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=16, num_blocks=3, dropout_prob=0.0):
        super(MLPModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.h = nn.ModuleList([MLPBlock(hidden_size, hidden_size, dropout_prob) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.dropout(x)
        for layer in self.h:
            x = layer(x)
        return x

class MLPForClassification(nn.Module):
    def __init__(self, input_size=16, hidden_size=16, num_classes=2, num_blocks=3, dropout_prob=0.0):
        super(MLPForClassification, self).__init__()
        self.mlp = MLPModel(input_size, hidden_size, num_blocks, dropout_prob)
        self.score = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.mlp(x)
        x = self.score(x)
        return x


def train_model(model,X_train,y_train,X_eval,y_eval,batch_size = 1024,epochs=3):
    # Create DataLoader
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.squeeze().long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model


def eval_test_model(model,X_data,y_data,batch_size = 1024,verbose=False):
    #Testing:
    test_dataset = TensorDataset(X_data, y_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch.squeeze()).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    model.train()
    if verbose:
        print(f"Test Accuracy: {accuracy:.4f}", flush=True)
    return accuracy

def train_model_early_stopping(model,X_train,y_train,X_eval,y_eval,batch_size = 1024,epochs=3,early_stopping_threshold=1024):
    
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    model.train()
    best_accuracy=-1
    no_improvement=0
    best_model=None
    early_stopping=False
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.squeeze().long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            eval_accuracy=eval_model(model,X_eval,y_eval,batch_size = 1024)
            if eval_accuracy>best_accuracy:
                best_accuracy=eval_accuracy
                no_improvement=0
                best_model = copy.deepcopy(model)
            elif no_improvement>=early_stopping_threshold:
                early_stopping=True
                break
            else:
                no_improvement+=1
        if early_stopping:
            break
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}","steps without improvement:",no_improvement,"best accuracy:",best_accuracy, flush=True)
    return best_model



def eval_model(model,X_eval,y_eval,batch_size = 1024,verbose=False):
    return eval_test_model(model,X_eval,y_eval,batch_size=batch_size,verbose=verbose)
    
    
def test_model(model,X_test,y_test,batch_size = 1024,verbose=False):
    return eval_test_model(model,X_test,y_test,batch_size=batch_size,verbose=verbose)


def make_model(X_train,y_train,X_eval,y_eval,X_test,y_test,epochs=3,device="cpu"):
    model = MLPForClassification()
    model.to(device)
    model=train_model_early_stopping(model,X_train,y_train,X_eval,y_eval,epochs=epochs)
    accuracy=test_model(model,X_test,y_test)
    return model,accuracy
