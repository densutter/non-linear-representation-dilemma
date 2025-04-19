#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import numpy as np
import copy
import json
from tqdm import tqdm


# In[2]:


from Helper_Functions import set_seed
from Dataset_Generation import Generate_LLM_Eval_Intervention_Data
from LLM_Model import (make_model,
                       LLM_Criterion)
from RevNet import RevNet
from Rotation_Model import Rotation
from DAS import phi_class
from DAS_LLM import Distributed_Alignment_Search_LLM


# In[3]:


model_config = {"model"   : "meta-llama/Llama-3.2-1B",
                "Trained" : False}

DEVICE        = "cpu" #"cuda:0" #"cuda"/"cpu"

"""
transformation_config = {"type"        : "Rotation",
                         "in_features" :       2048}
"""
transformation_config = {"type"          : "RevNet",
                         "number_blocks" :       10,
                         "in_features"   :     2048,
                         "hidden_size"   :     2048*2}

Max_Epochs                       = 10 #4 #1 #50
Early_Stopping_Epochs            = 10 #4 #1 #50
early_stopping_improve_threshold = 0.001
LLM_test_samples                 = 1600
Intervention_train_size          = 32000*4 #1280000
Intervention_eval_size           = 1600
Intervention_test_size           = 1600
learning_rate                    = 0.000001


# In[4]:


LLM_test_data,DAS_Train,DAS_Eval,DAS_Test=Generate_LLM_Eval_Intervention_Data(filename='./mecha_ioi_200k.parquet',
                                                                              model_name=model_config["model"],
                                                                              LLM_test_samples=LLM_test_samples,
                                                                              Intervention_train_size=Intervention_train_size,
                                                                              Intervention_eval_size=Intervention_eval_size,
                                                                              Intervention_test_size=Intervention_test_size)


# In[ ]:


results=[]
for acseed in [4287]:
    results.append({})
    set_seed(acseed)
    model,accuracy=make_model(model_config["model"],LLM_test_data,model_config["Trained"],device=DEVICE)
    Layers=[]
    #Layers.append(("Layer7",model.model.layers[7]))
    Layers.append(("Layer9",model.model.layers[9]))
    #Layers.append(("Layer14",model.model.layers[14]))
    inter_dims=[]
    inter_dims.append([list(range(0,transformation_config["in_features"]//2))])
    #inter_dims.append([list(range(0,transformation_config["in_features"]//64))])
    #inter_dims.append([list(range(0,1))])
    
    results[-1]["accuracy"]=accuracy
    for LayerName,Layer in Layers:
        results[-1][LayerName]={}
        for inter_dim in inter_dims:
            print(LayerName,":",len(inter_dim[0]), flush=True)
            #Initialize transformation function
            
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
            optimizer = optim.Adam(p.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
            criterion = LLM_Criterion
            
            
            phi=phi_class(p,p_inverse,LLM_Criterion,optimizer,scheduler)

            
    
            DAS_Experiment=Distributed_Alignment_Search_LLM(Model=model,
                                                            Model_Layer=Layer,
                                                            Train_Data_Raw=DAS_Train,
                                                            Test_Data_Raw=DAS_Test,
                                                            Eval_Data_Raw=DAS_Eval,
                                                            Hidden_Layer_Size=transformation_config["in_features"],
                                                            Variable_Dimensions=inter_dim,
                                                            Transformation_Class=phi,
                                                            Device=DEVICE)
    
            DAS_Experiment.train_test(batch_size=32,
                                      epochs=Max_Epochs,
                                      mode=1,
                                      early_stopping_threshold=Early_Stopping_Epochs,
                                      early_stopping_improve_threshold=early_stopping_improve_threshold,
                                      verbose=True) #Train
    
            accuracy=DAS_Experiment.train_test(batch_size=32,
                                               mode=2,
                                               verbose=True)#Test
            
            results[-1][LayerName][str(inter_dim)]=accuracy
            DAS_Experiment.Cleanup()
            DAS_Experiment=None
            with open('results.json', 'w') as f:
                json.dump(results, f)

