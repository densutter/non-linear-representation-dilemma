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


# In[2]:


import wandb
wandb.init(mode="disabled")


# In[3]:


from das.Classification_Model import make_model_DAS_fitted
from das.Helper_Functions import *
from das.Dataset_Generation import (make_intervention_dataset_AndOrAnd,
                                make_intervention_dataset_AndOr,
                                make_intervention_dataset_AndOrAnd_DAS_Fitted,
                                make_intervention_dataset_AndOr_DAS_Fitted)
from das.RevNet import RevNet
from das.Rotation_Model import Rotation
from das.DAS import phi_class
from das.DAS_MLP import Distributed_Alignment_Search_MLP


# In[4]:


#For different Settings and transformation functions please adapt this configurations:

DEVICE  = sys.argv[1]#"cuda"/"cpu"
Setting  = sys.argv[2]
#Setting = "AndOrAnd"
#Setting = "AndOr"

FitModelTo = sys.argv[3]
#FitModelTo = "AndOrAnd"
#FitModelTo = "AndOr"


if sys.argv[4]=="Rotation":
    transformation_config = {"type"        : "Rotation",
                             "in_features" :         24}
elif sys.argv[4]=="RevNet":
    transformation_config = {"type"          : "RevNet",
                             "number_blocks" :       10,
                             "in_features"   :       24,
                             "hidden_size"   :       24}


Max_Epochs                       = 50
Early_Stopping_Epochs            = 5
early_stopping_improve_threshold = 0.001
ReduceLROnPlateau_patience       = 10

ensure_directory_for_file(sys.argv[5])


# In[ ]:


results=[]
for acseed in [4287, 3837, 9097, 2635, 5137, 6442, 5234, 4641, 8039, 2266]:
    results.append({})
    set_seed(acseed)
    if FitModelTo == "AndOrAnd":
        DAS_Train = make_intervention_dataset_AndOrAnd_DAS_Fitted(1280000,4)
        DAS_Test  = make_intervention_dataset_AndOrAnd_DAS_Fitted(10000,4)
        DAS_Eval  = make_intervention_dataset_AndOrAnd_DAS_Fitted(10000,4)
    elif FitModelTo == "AndOr":
        DAS_Train = make_intervention_dataset_AndOr_DAS_Fitted(1280000,4)
        DAS_Test  = make_intervention_dataset_AndOr_DAS_Fitted(10000,4)
        DAS_Eval  = make_intervention_dataset_AndOr_DAS_Fitted(10000,4)
    
    model,accuracy = make_model_DAS_fitted(DAS_Train=DAS_Train,
                                           DAS_Test=DAS_Test,
                                           DAS_Eval=DAS_Eval,
                                           Hidden_Layer_Size=24,
                                           inter_dim=[list(range(0,12)),list(range(12,24))],
                                           DEVICE=DEVICE,
                                           Max_Epochs=50,
                                           Early_Stopping_Epochs=5,
                                           early_stopping_improve_threshold=0.001,
                                           ReduceLROnPlateau_patience=10)

        
    Layers=[]
    Layers.append(("Layer1",model.mlp.h[0]))
    Layers.append(("Layer2",model.mlp.h[1]))
    Layers.append(("Layer3",model.mlp.h[2]))
    inter_dims=[]
    
    inter_dims.append([list(range(0,transformation_config["in_features"]//2)),list(range(transformation_config["in_features"]//2,transformation_config["in_features"]))])
    inter_dims.append([list(range(0,2)),list(range(2,4))])
    inter_dims.append([list(range(0,1)),list(range(1,2))])
    



    if Setting == "AndOrAnd":
        DAS_Train = make_intervention_dataset_AndOrAnd(1280000,4)
        DAS_Test  = make_intervention_dataset_AndOrAnd(10000,4)
        DAS_Eval  = make_intervention_dataset_AndOrAnd(10000,4)
    elif Setting == "AndOr":
        DAS_Train = make_intervention_dataset_AndOr(1280000,4)
        DAS_Test  = make_intervention_dataset_AndOr(10000,4)
        DAS_Eval  = make_intervention_dataset_AndOr(10000,4)
    else:
        Exception("Unknown Setting")
        
    results[-1]["accuracy"]=accuracy
    for LayerName,Layer in Layers:
        results[-1][LayerName]={}
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

            
    
            DAS_Experiment=Distributed_Alignment_Search_MLP(Model=model,
                                                            Model_Layer=Layer,
                                                            Train_Data_Raw=DAS_Train,
                                                            Test_Data_Raw=DAS_Test,
                                                            Eval_Data_Raw=DAS_Eval,
                                                            Hidden_Layer_Size=transformation_config["in_features"],
                                                            Variable_Dimensions=inter_dim,
                                                            Transformation_Class=phi,
                                                            Device=DEVICE)
    
            DAS_Experiment.train_test(batch_size=6400,
                                      epochs=Max_Epochs,
                                      mode=1,
                                      early_stopping_threshold=Early_Stopping_Epochs,
                                      early_stopping_improve_threshold=early_stopping_improve_threshold) #Train
    
            accuracy=DAS_Experiment.train_test(batch_size=6400,
                                               mode=2)#Test
            
            results[-1][LayerName][str(inter_dim)]=accuracy
            DAS_Experiment.Cleanup()
            DAS_Experiment=None
            with open(sys.argv[5], 'w') as f:
                json.dump(results, f)


# In[ ]:




