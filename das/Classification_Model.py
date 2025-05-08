import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import copy

from .DAS import phi_class
from .DAS_MLP import Distributed_Alignment_Search_MLP

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
    def __init__(self, input_size=16, num_blocks=3, dropout_prob=0.0):
        super(MLPModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.h = nn.ModuleList([MLPBlock(input_size, input_size, dropout_prob) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.dropout(x)
        for layer in self.h:
            x = layer(x)
        return x

class MLPForClassification(nn.Module):
    def __init__(self, input_size=16, num_classes=2, num_blocks=3, dropout_prob=0.0):
        super(MLPForClassification, self).__init__()
        self.mlp = MLPModel(input_size, num_blocks, dropout_prob)
        self.score = nn.Linear(input_size, num_classes)

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


def make_model(X_train,y_train,X_eval,y_eval,X_test,y_test,input_size,epochs=3,device="cpu"):
    model = MLPForClassification(input_size=input_size)
    model.to(device)
    model=train_model_early_stopping(model,X_train,y_train,X_eval,y_eval,epochs=epochs)
    accuracy=test_model(model,X_test,y_test)
    return model,accuracy





###############################
##                           ##
##  For DAS fitted Training  ##
##                           ##
###############################




class IdentityModel(nn.Module):
    def __init__(self,params):
        super(IdentityModel, self).__init__()
        self.params=list(params)

    def forward(self, x):
        return x   
    
    def inverse(self, x):
        return x

    def parameters(self):
        return self.params
        


def train_model_DAS_fitting(DAS_Experiment,
                            Max_Epochs,
                            Early_Stopping_Epochs,
                            early_stopping_improve_threshold):


    DAS_Experiment.train_test(batch_size=6400,
                              epochs=Max_Epochs,
                              mode=1,
                              early_stopping_threshold=Early_Stopping_Epochs,
                              early_stopping_improve_threshold=early_stopping_improve_threshold,
                              TrainModel=True,
                              verbose=True) #Train

    return DAS_Experiment

def test_model_DAS_fitting(DAS_Experiment):
    accuracy=DAS_Experiment.train_test(batch_size=6400,
                                       mode=2)#Test
    return accuracy


def make_model_DAS_fitted(DAS_Train,
                          DAS_Test,
                          DAS_Eval,
                          Hidden_Layer_Size,
                          inter_dim,
                          DEVICE,
                          Max_Epochs,
                          Early_Stopping_Epochs,
                          early_stopping_improve_threshold,
                          ReduceLROnPlateau_patience):
    
    model = MLPForClassification(input_size=Hidden_Layer_Size)
    model.to(DEVICE)
    Layer = model.mlp.h[1]
    
    p         = IdentityModel(model.parameters())
    p_inverse = p.inverse
    optimizer = optim.Adam(p.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=ReduceLROnPlateau_patience)
    criterion = nn.CrossEntropyLoss()
    phi=phi_class(p,p_inverse,criterion,optimizer,scheduler)
    
    DAS_Experiment=Distributed_Alignment_Search_MLP(Model=model,
                                                Model_Layer=Layer,
                                                Train_Data_Raw=DAS_Train,
                                                Test_Data_Raw=DAS_Test,
                                                Eval_Data_Raw=DAS_Eval,
                                                Hidden_Layer_Size=Hidden_Layer_Size,
                                                Variable_Dimensions=inter_dim,
                                                Transformation_Class=phi,
                                                Device=DEVICE)
    DAS_Experiment=train_model_DAS_fitting(DAS_Experiment,
                                           Max_Epochs,
                                           Early_Stopping_Epochs,
                                           early_stopping_improve_threshold)
    accuracy=test_model_DAS_fitting(DAS_Experiment)
    model=DAS_Experiment.Model
    DAS_Experiment.Cleanup()
    DAS_Experiment=None
    return model,accuracy
