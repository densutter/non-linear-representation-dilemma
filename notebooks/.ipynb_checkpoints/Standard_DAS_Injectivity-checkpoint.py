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


# In[3]:


def analyze_layer_distances(Layer_Save, sample_size=1000, device='cuda'):
    # Convert to torch tensor
    feature_layers = torch.stack([torch.tensor(arr) for arr in Layer_Save[:-3]]).to(device)  # [num_layers, num_points, dim]
    os_labels = torch.tensor(Layer_Save[-3]).to(device)
    vs_labels = torch.stack([torch.tensor(Layer_Save[-2]), torch.tensor(Layer_Save[-1])], dim=1).to(device)

    num_layers, num_points, dim = feature_layers.shape

    # Sample indices
    all_indices = list(range(num_points))
    sampled = random.sample(all_indices, sample_size)
    sampled_tensor = torch.tensor(sampled, device=device)

    # Initialize result structure
    categories = ["AP", "OS", "VS", "nOS", "nVS"]
    Results = [
        {key: torch.tensor([0, 0.0, float('inf')], device=device) for key in categories}
        for _ in range(num_layers)
    ]

    for i_h, i in enumerate(tqdm(sampled_tensor.tolist(), desc="Processing samples")):
        exclude_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        exclude_mask[sampled_tensor[:i_h+1]] = True

        for k in range(num_layers):
            xi = feature_layers[k, i]  # [dim]
            xj = feature_layers[k]     # [num_points, dim]

            # Euclidean distances (broadcasted)
            dists = torch.norm(xj - xi, dim=1)  # [num_points]
            dists = dists.masked_fill(exclude_mask, float('inf'))

            # Label masks
            os_mask = (os_labels == os_labels[i]) & ~exclude_mask
            vs_mask = ((vs_labels[:, 0] == vs_labels[i, 0]) &
                       (vs_labels[:, 1] == vs_labels[i, 1]) &
                       ~exclude_mask)

            # Inverse masks (excluding masked-out/self points)
            nos_mask = ~os_mask & ~exclude_mask
            nvs_mask = ~vs_mask & ~exclude_mask

            # Collect all masks
            mask_dict = {
                "AP": ~exclude_mask,
                "OS": os_mask,
                "VS": vs_mask,
                "nOS": nos_mask,
                "nVS": nvs_mask
            }

            # Aggregate results
            for key, mask in mask_dict.items():
                valid_dists = dists[mask]
                if valid_dists.numel() == 0:
                    continue
                Results[k][key][0] += valid_dists.numel()
                Results[k][key][1] += valid_dists.sum()
                Results[k][key][2] = torch.minimum(Results[k][key][2], valid_dists.min())

    # Convert to CPU + readable format
    final_results = [
        {k: v.cpu().numpy().tolist() for k, v in layer_dict.items()}
        for layer_dict in Results
    ]

    return final_results


# In[4]:


DEVICE  = sys.argv[1]#"cuda"/"cpu"
num_classes=2

ensure_directory_for_file(sys.argv[2])

def register_intervention_hook(Save_array,Pos,layer):
    def hook_fn(module, input, output):
        Save_array[Pos].append(output.detach().cpu())
    layer.register_forward_hook(hook_fn)


Full_results=[]
for acseed in [4287, 3837, 9097, 2635, 5137, 6442, 5234, 4641, 8039, 2266]:
    set_seed(acseed)
    X_train,y_train = make_model_dataset(1048576,4,DEVICE)#1048576
    X_eval,y_eval   = make_model_dataset(10000,4,DEVICE)#10000
    X_test,y_test   = make_model_dataset(10000,4,DEVICE)
   
    model,accuracy=make_model(X_train,y_train,X_eval,y_eval,X_test,y_test,input_size=16,epochs=20,device=DEVICE)
    Layers=[]
    Layers.append(("Layer1",model.mlp.h[0]))
    Layers.append(("Layer2",model.mlp.h[1]))
    Layers.append(("Layer3",model.mlp.h[2]))


    Layer_Save=[[],[],[],[]]
    Hooks=[]
    predicted_classes_set = set()
    for acpos,aclayer in enumerate(Layers):
        Layer_Save.append([])
        Hooks.append(register_intervention_hook(Layer_Save,acpos+1,aclayer[1]))

    
    X_train,y_train = make_model_dataset(1280000,4,DEVICE)
    test_dataset = TensorDataset(X_train,y_train)
    test_loader = DataLoader(test_dataset, batch_size=6400, shuffle=False)#6400
    
    
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
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

    Results=[]
    for _ in range(len(Layer_Save)-3):
        Results.append({})
        Results[-1]["Overlap"]=None
        Results[-1]["AP"]=[0,0,np.inf]
        Results[-1]["OS"]=[0,0,np.inf]
        Results[-1]["nOS"]=[0,0,np.inf]
        Results[-1]["VS"]=[0,0,np.inf]
        Results[-1]["nVS"]=[0,0,np.inf]
        
    Results=analyze_layer_distances(Layer_Save, sample_size=10000, device='cuda')
    
    for i in range(len(Layer_Save)-3):
        seen = set()
        duplicates = []
        
        for point in tqdm(map(tuple, Layer_Save[i])):
            if point in seen:
                duplicates.append(point)
            else:
                seen.add(point)
        Results[i]["Overlap"]=len(duplicates)

    Full_results.append(Results)
    
    with open(sys.argv[2], 'w') as f:
        json.dump(Full_results, f)

