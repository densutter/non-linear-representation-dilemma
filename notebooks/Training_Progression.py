#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import wandb
wandb.init(mode="disabled")


# In[ ]:


from das.Classification_Model import (MLPForClassification,
                                  train_model,
                                  eval_model,
                                  test_model)
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


# In[ ]:


#For different Settings and transformation functions please adapt this configurations:

DEVICE  = sys.argv[1]#"cuda"/"cpu"
Setting  = sys.argv[2]
#Setting = "Both Equality Relations"
#Setting = "Left Equality Relation"
#Setting = "Identity of First Argument"
#Setting = "AndOrAnd"
#Setting = "AndOr"
hid_size=16
if Setting=="AndOrAnd" or "AndOr":
    hid_size=24

if sys.argv[3]=="Rotation":
    transformation_config = {"type"        : "Rotation",
                             "in_features" :   hid_size} #24 for the Settings "AndOrAnd" and "AndOr"
elif sys.argv[3]=="RevNet":
    transformation_config = {"type"          :         "RevNet",
                             "number_blocks" : int(sys.argv[4]),
                             "in_features"   :         hid_size, #24 for the Settings "AndOrAnd" and "AndOr"
                             "hidden_size"   : int(sys.argv[5])} 

Max_Epochs                       = 50
Early_Stopping_Epochs            = 5
early_stopping_improve_threshold = 0.001
ReduceLROnPlateau_patience       = 10


ensure_directory_for_file(sys.argv[6])    
# In[ ]:


one_variable_settings = ["Identity of First Argument"]
two_variable_settings = ["Both Equality Relations","Left Equality Relation","AndOrAnd","AndOr"]
DAS_Original_tasks    = ["Both Equality Relations","Left Equality Relation","Identity of First Argument"]
AndOrAnd_tasks        = ["AndOrAnd","AndOr"]


# In[ ]:


#Helper Functions:

def chunk_list(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


# In[ ]:


#DAS Training Loop

results=[]
for acseed in [4287, 3837, 9097, 2635, 5137]:
    set_seed(acseed)
    model = MLPForClassification(input_size=transformation_config["in_features"])
    model.to(DEVICE)
    if Setting in DAS_Original_tasks:
        X_train,y_train = make_model_dataset(1048576,4,DEVICE)
        X_eval,y_eval   = make_model_dataset(10000,4,DEVICE)
        X_test,y_test   = make_model_dataset(10000,4,DEVICE)
    elif Setting in AndOrAnd_tasks:
        X_train,y_train = make_model_dataset_AndOrAnd(1048576,4,DEVICE)
        X_eval,y_eval   = make_model_dataset_AndOrAnd(10000,4,DEVICE)
        X_test,y_test   = make_model_dataset_AndOrAnd(10000,4,DEVICE)
        

    indexes=[None]
    indexes_num=list(range(X_train.shape[0]))
    random.shuffle(indexes_num)
    indexes+=chunk_list(indexes_num, 1048576//8)
    random.shuffle(indexes_num)
    indexes+=chunk_list(indexes_num, 1048576//8)
    
    Layers=[]
    Layers.append(("Layer1",model.mlp.h[0]))
    Layers.append(("Layer2",model.mlp.h[1]))
    Layers.append(("Layer3",model.mlp.h[2]))
    inter_dims=[]

    if Setting in two_variable_settings:
        inter_dims.append([list(range(0,transformation_config["in_features"]//2)),list(range(transformation_config["in_features"]//2,transformation_config["in_features"]))])
        inter_dims.append([list(range(0,2)),list(range(2,4))])
        inter_dims.append([list(range(0,1)),list(range(1,2))])
    elif Setting in one_variable_settings:
        inter_dims.append([list(range(0,transformation_config["in_features"]//2))])
        inter_dims.append([list(range(0,2))])
        inter_dims.append([list(range(0,1))])
    else:
        Exception("Unknown Setting")

    if Setting == "Both Equality Relations":
        DAS_Train = make_intervention_dataset_variable_intervention_all(1280000,4)
        DAS_Test  = make_intervention_dataset_variable_intervention_all(10000,4)
        DAS_Eval  = make_intervention_dataset_variable_intervention_all(10000,4)
    elif Setting == "Left Equality Relation":
        DAS_Train = make_intervention_dataset_variable_intervention_first(1280000,4)
        DAS_Test  = make_intervention_dataset_variable_intervention_first(10000,4)
        DAS_Eval  = make_intervention_dataset_variable_intervention_first(10000,4)
    elif Setting == "Identity of First Argument":
        DAS_Train = make_intervention_dataset_first_input_intervention(1280000,4)
        DAS_Test  = make_intervention_dataset_first_input_intervention(10000,4)
        DAS_Eval  = make_intervention_dataset_first_input_intervention(10000,4)
    elif Setting == "AndOrAnd":
        DAS_Train = make_intervention_dataset_AndOrAnd(1280000,4)
        DAS_Test  = make_intervention_dataset_AndOrAnd(10000,4)
        DAS_Eval  = make_intervention_dataset_AndOrAnd(10000,4)
    elif Setting == "AndOr":
        DAS_Train = make_intervention_dataset_AndOr(1280000,4)
        DAS_Test  = make_intervention_dataset_AndOr(10000,4)
        DAS_Eval  = make_intervention_dataset_AndOr(10000,4)
    else:
        Exception("Unknown Setting")

    results.append([])
    for ac_in in indexes:   
        results[-1].append({})
        model.train()
        for param in model.parameters():
            param.requires_grad = True  # This unfreezes the weights
        if ac_in is not None:# first DAS on untrained
            model=train_model(model,X_train[ac_in],y_train[ac_in],X_eval,y_eval,batch_size = 1024,epochs=1)
        accuracy=test_model(model,X_test,y_test)
        
        results[-1][-1]["accuracy"]=accuracy
        for LayerName,Layer in Layers:
            results[-1][-1][LayerName]={}
            for inter_dim in inter_dims:
                print(LayerName,":",inter_dim, flush=True)
        
                #Initialize transformation function
                if transformation_config["type"]=="Rotation":
                    p = Rotation(transformation_config["in_features"])
                elif transformation_config["type"]=="RevNet":
                    p = RevNet(number_blocks =  transformation_config["number_blocks"],
                               in_features   =  transformation_config["in_features"],
                               hidden_size   =  transformation_config["hidden_size"]
                              )
                else:
                    Exception("Unknown transformation function")
                p.to(DEVICE)
                p_inverse = p.inverse
                optimizer = optim.Adam(p.parameters(), lr=0.001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=ReduceLROnPlateau_patience)
                criterion = nn.CrossEntropyLoss()
                phi=phi_class(p,p_inverse,criterion,optimizer,scheduler)
        
                
        
                DAS_Experiment=Distributed_Alignment_Search_MLP(Model                = model,
                                                                Model_Layer          = Layer,
                                                                Train_Data_Raw       = DAS_Train,
                                                                Test_Data_Raw        = DAS_Test,
                                                                Eval_Data_Raw        = DAS_Eval,
                                                                Hidden_Layer_Size    = transformation_config["in_features"],
                                                                Variable_Dimensions  = inter_dim,
                                                                Transformation_Class = phi,
                                                                Device               = DEVICE)
        
                DAS_Experiment.train_test(batch_size=6400,
                                          epochs=Max_Epochs,
                                          mode=1,
                                          early_stopping_threshold=Early_Stopping_Epochs,
                                          early_stopping_improve_threshold=early_stopping_improve_threshold) #Train
        
                accuracy=DAS_Experiment.train_test(batch_size=6400,
                                                   mode=2)
                
                results[-1][-1][LayerName][str(inter_dim)]=accuracy
                DAS_Experiment.Cleanup()
                DAS_Experiment=None
                with open(sys.argv[6], 'w') as f:
                    json.dump(results, f)

