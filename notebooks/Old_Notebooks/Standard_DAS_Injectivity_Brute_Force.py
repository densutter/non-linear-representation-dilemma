#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import numpy as np
import copy
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# In[2]:


import wandb
wandb.init(mode="disabled")


# In[3]:


from das.Classification_Model import (MLPForClassification,
                                  train_model,
                                  eval_model,
                                  test_model,
                                  make_model)
from das.Helper_Functions import *
from das.Dataset_Generation import (make_model_dataset,
                                make_model_dataset_AndOrAnd,
                                make_intervention_dataset_variable_intervention_all,
                                make_intervention_dataset_variable_intervention_first,
                                make_intervention_dataset_first_input_intervention,
                                make_intervention_dataset_AndOrAnd,
                                make_intervention_dataset_AndOr)
from das.RevNet import RevNet
from das.Rotation_Model import Rotation
from das.DAS import phi_class
from das.DAS_MLP import Distributed_Alignment_Search_MLP


# In[4]:


DEVICE  = sys.argv[1]#"cuda"/"cpu"
num_classes=2

ensure_directory_for_file(sys.argv[2])

# In[5]:


def register_intervention_hook(Save_array,Pos,layer):
    def hook_fn(module, input, output):
        Save_array[Pos].append(output.detach().cpu())
    layer.register_forward_hook(hook_fn)


# In[6]:


Full_results=[]
for acseed in [4287, 3837, 9097, 2635, 5137, 6442, 5234, 4641, 8039, 2266]:
    set_seed(acseed)
    X_train,y_train = make_model_dataset(1048576,4,DEVICE)#1048576
    X_eval,y_eval   = make_model_dataset(10000,4,DEVICE)#10000
    X_test,y_test   = make_model_dataset(10000,4,DEVICE)
    X_inj,y_inj   = make_model_dataset(10000,4,DEVICE)#5000
   
    model,accuracy=make_model(X_train,y_train,X_eval,y_eval,X_test,y_test,input_size=16,epochs=20,device=DEVICE)
    Layers=[]
    Layers.append(("Layer1",model.mlp.h[0]))
    Layers.append(("Layer2",model.mlp.h[1]))
    Layers.append(("Layer3",model.mlp.h[2]))


    Layer_Save=[[],[],[],[]]
    Results=[]
    Results.append([0,0,0,0,0,0,0])
    Hooks=[]
    predicted_classes_set = set()
    for acpos,aclayer in enumerate(Layers):
        Layer_Save.append([])
        Results.append([0,0,0,0,0,0,0])
        Hooks.append(register_intervention_hook(Layer_Save,acpos+1,aclayer[1]))
    
    
    test_dataset = TensorDataset(X_inj,y_inj)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)#6400
    
    
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            Layer_Save[0].append(X_batch)
            output=model(X_batch)
            predicted = torch.argmax(output, dim=1)
            Layer_Save[-3].append(predicted)
            Layer_Save[-2].append(torch.all(X_batch[:, :4] == X_batch[:, 4:8], dim=1))
            Layer_Save[-1].append(torch.all(X_batch[:, 8:12] == X_batch[:, 12:16], dim=1))
        
            # Update set with predictions from this batch
            predicted_classes_set.update(predicted.tolist())
    for i in range(len(Layer_Save)):
        Layer_Save[i]=torch.cat(Layer_Save[i])
    for i in tqdm(range(len(Layer_Save[0]))):
        for j in range(i+1,len(Layer_Save[0])):
            Results[0][0]+=1
            Results[0][1]+=torch.norm(Layer_Save[0][i]-Layer_Save[0][j], p=2).item()
            if Layer_Save[-3][i]==Layer_Save[-3][j]:
                Results[0][3]+=1
                Results[0][4]+=torch.norm(Layer_Save[0][i]-Layer_Save[0][j], p=2).item()
            if Layer_Save[-2][i]==Layer_Save[-2][j] and Layer_Save[-1][i]==Layer_Save[-1][j]:
                Results[0][5]+=1
                Results[0][6]+=torch.norm(Layer_Save[0][i]-Layer_Save[0][j], p=2).item()
            if torch.equal(Layer_Save[0][i],Layer_Save[0][j]):
                Results[0][2]+=1
            for k in range(1,len(Layer_Save)-3):
                Results[k][0]+=1
                Results[k][1]+=torch.norm(Layer_Save[k][i]-Layer_Save[k][j], p=2).item()
                if torch.equal(Layer_Save[k][i],Layer_Save[k][j]):
                    Results[k][2]+=1
                if Layer_Save[-3][i]==Layer_Save[-3][j]:
                    Results[k][3]+=1
                    Results[k][4]+=torch.norm(Layer_Save[k][i]-Layer_Save[k][j], p=2).item()
                if Layer_Save[-2][i]==Layer_Save[-2][j] and Layer_Save[-1][i]==Layer_Save[-1][j]:
                    Results[k][5]+=1
                    Results[k][6]+=torch.norm(Layer_Save[k][i]-Layer_Save[k][j], p=2).item()
    
    all_classes_set = set(range(num_classes))
    missing_classes_set = all_classes_set - predicted_classes_set
    Results_processed=[]
    for i in Results:
        Results_processed.append([i[1]/i[0],i[2]/i[0],i[4]/i[3],i[6]/i[5]])
    print("Surjectivity: Missing:", missing_classes_set,"Found:",predicted_classes_set)
    print("Injectivity", Results_processed)  
    print("Injectivity", Results) 
    Full_results.append({})
    Full_results[-1]["Results"]=Results
    Full_results[-1]["Results_processed"]=Results_processed
    Full_results[-1]["missing_classes_set"]=list(missing_classes_set)
    Full_results[-1]["predicted_classes_set"]=list(predicted_classes_set)
    with open(sys.argv[2], 'w') as f:
        json.dump(Full_results, f)






