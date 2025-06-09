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


from das.Classification_Model import (MLPForClassification,
                                      make_model_DAS_fitted)
from das.Helper_Functions import *
from das.Dataset_Generation import (make_intervention_dataset_AndOrAnd_DAS_Fitted,
                                    make_intervention_dataset_AndOr_DAS_Fitted,
                                    make_intervention_dataset_AndOrAnd,
                                    make_intervention_dataset_AndOr)
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
hid_size=16
if Setting=="AndOrAnd" or Setting=="AndOr":
    hid_size=24


transformation_config = {"type"        : "identity",
                         "in_features" :   hid_size} #24 for the Settings "AndOrAnd" and "AndOr"

Max_Epochs                       = 50
Early_Stopping_Epochs            = 5
early_stopping_improve_threshold = 0.001
ReduceLROnPlateau_patience       = 10

ensure_directory_for_file(sys.argv[4])


# In[5]:


one_variable_settings = ["Identity of First Argument","Left Equality Relation"]
two_variable_settings = ["Both Equality Relations","AndOrAnd","AndOr"]
DAS_Original_tasks    = ["Both Equality Relations","Left Equality Relation","Identity of First Argument"]
AndOrAnd_tasks        = ["AndOrAnd","AndOr"]
numvar=None
if Setting in two_variable_settings:
    numvar=2
elif Setting in one_variable_settings:
    numvar=1


# In[6]:


def Process_Data_Left_Equality(Data):
    for i in range(len(Data)):
        Data[i]["sources"]=Data[i]["sources"][:1]
        Data[i]["intervention"]=Data[i]["intervention"][:1]
    return Data


# In[7]:


def make_all_pairs(possible_list, used_list):
    pairs = []
    
    for i in possible_list:
        if i in used_list:
            continue
        for j in possible_list:
            if j in used_list:
                continue
            if i==j:
                continue
            pairs.append([i,j])
    
    return pairs


# In[8]:


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
    

    if Setting == "AndOrAnd":
        DAS_Train = make_intervention_dataset_AndOrAnd(1,4)
        DAS_Test  = make_intervention_dataset_AndOrAnd(6400,4)
        DAS_Eval  = make_intervention_dataset_AndOrAnd(6400,4)
    elif Setting == "AndOr":
        DAS_Train = make_intervention_dataset_AndOr(1,4)
        DAS_Test  = make_intervention_dataset_AndOr(6400,4)
        DAS_Eval  = make_intervention_dataset_AndOr(6400,4)
    else:
        Exception("Unknown Setting")
        
    results[-1]["accuracy"]=accuracy
    for LayerName,Layer in Layers:
        results[-1][LayerName]={}
            
    
        p = torch.nn.Identity()
        p.to(DEVICE)
        p_inverse = torch.nn.Identity()
        optimizer = None #optim.Adam(p.parameters(), lr=0.001)
        scheduler = None #optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=ReduceLROnPlateau_patience)
        criterion = nn.CrossEntropyLoss()
        
        
        phi=phi_class(p,p_inverse,criterion,optimizer,scheduler)




        if Setting in two_variable_settings:
            neuronset=[[],[]]
            results[-1][LayerName]=[]
            while len(neuronset[0])<transformation_config["in_features"]//2:
                pairs=make_all_pairs(list(range(transformation_config["in_features"])),neuronset[0]+neuronset[1])
                best_pair_neurons=None
                best_pair_acc=-1
                best_pair_loss=None
                for pair in pairs:
                    ac_neuronset=copy.deepcopy(neuronset)
                    ac_neuronset[0].append(pair[0])
                    ac_neuronset[1].append(pair[1])
                    DAS_Experiment=Distributed_Alignment_Search_MLP(Model=model,
                                                                    Model_Layer=Layer,
                                                                    Train_Data_Raw=DAS_Train,
                                                                    Test_Data_Raw=DAS_Eval,
                                                                    Eval_Data_Raw=DAS_Train,
                                                                    Hidden_Layer_Size=transformation_config["in_features"],
                                                                    Variable_Dimensions=ac_neuronset,
                                                                    Transformation_Class=phi,
                                                                    Device=DEVICE)
                    _,Dloss=DAS_Experiment.train_test(batch_size=6400,
                                                       mode=2,
                                                       report_loss=True)#Test
                    DAS_Experiment.Test_Dataset,DAS_Experiment.Test_Samples_Number=DAS_Experiment.Prepare_Dataset(DAS_Test)
                    accur=DAS_Experiment.train_test(batch_size=6400,
                                                        mode=2,)#Test
                    DAS_Experiment.Cleanup()
                    if (best_pair_loss is None) or best_pair_loss>Dloss:
                        best_pair_loss=Dloss
                        best_pair_acc=accur
                        best_pair_neurons=ac_neuronset
                results[-1][LayerName].append((best_pair_neurons,best_pair_acc))
                neuronset=copy.deepcopy(best_pair_neurons)
        if Setting in one_variable_settings:
            neuronset=[[]]
            results[-1][LayerName]=[]
            while len(neuronset[0])<transformation_config["in_features"]//2:
                best_pair_neurons=None
                best_pair_acc=-1
                for element in list(range(transformation_config["in_features"])):
                    if element in neuronset[0]:
                        continue
                    ac_neuronset=copy.deepcopy(neuronset)
                    ac_neuronset[0].append(element)
                    DAS_Experiment=Distributed_Alignment_Search_MLP(Model=model,
                                                                    Model_Layer=Layer,
                                                                    Train_Data_Raw=DAS_Train,
                                                                    Test_Data_Raw=DAS_Eval,
                                                                    Eval_Data_Raw=DAS_Train,
                                                                    Hidden_Layer_Size=transformation_config["in_features"],
                                                                    Variable_Dimensions=ac_neuronset,
                                                                    Transformation_Class=phi,
                                                                    Device=DEVICE)
                    _,Dloss=DAS_Experiment.train_test(batch_size=6400,
                                                       mode=2,
                                                       report_loss=True)
                    DAS_Experiment.Test_Dataset,DAS_Experiment.Test_Samples_Number=DAS_Experiment.Prepare_Dataset(DAS_Test)
                    accur=DAS_Experiment.train_test(batch_size=6400,
                                                        mode=2,)#Test
                    DAS_Experiment.Cleanup()
                    if (best_pair_loss is None) or best_pair_loss>Dloss:
                        best_pair_loss=Dloss
                        best_pair_acc=accur
                        best_pair_neurons=ac_neuronset
                results[-1][LayerName].append((best_pair_neurons,best_pair_acc))
                neuronset=copy.deepcopy(best_pair_neurons)

        
        DAS_Experiment=None
        with open(sys.argv[4], 'w') as f:
            json.dump(results, f)


# In[ ]:




