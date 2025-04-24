from transformers import AutoModelForCausalLM, AutoConfig
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import math
from tqdm import tqdm


def test_model(model,LLM_test_data,batch_size = 32,device="cpu"):
    #Testing:
    X_test,y_test = LLM_test_data
    
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for dp in tqdm(range(len(X_test))):
            ini=X_test[dp].to(device)
            outputs = model(**ini)["logits"][:,-1]
            true_logits=outputs[0][y_test[dp][0]]
            false_logits=outputs[0][y_test[dp][1]]
            if true_logits>false_logits:
                correct += 1
            total += 1
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}", flush=True)
    return accuracy



def make_model(model_name,LLM_test_data,Trained=True,device="cpu"): #Trained=0:pretrained, 1:fully randomized, 2:only randomize llm head
    if Trained==0:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    elif Trained==1:
        config = AutoConfig.from_pretrained(model_name)
        config.torch_dtype="float32"
        model = AutoModelForCausalLM.from_config(config).to(device)
    elif Trained==2:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        #print(model.model.embed_tokens.weight)
        #print(model.lm_head.weight)
        model.lm_head.weight=nn.Parameter(model.lm_head.weight.clone())
        init.kaiming_uniform_(model.lm_head.weight, a=math.sqrt(5)) 
        #print(model.model.embed_tokens.weight)
        #print(model.lm_head.weight)
    else:
        Exception("Unknown initialization")
    accuracy=test_model(model,LLM_test_data,device=device)
    return model,accuracy



def LLM_Criterion_Diff(false_logits, true_logits):
    diff = false_logits - true_logits
    #score = torch.exp(diff)
    #print(diff)
    #print(torch.where(diff >= 0, diff, 0.1 * diff))
    return torch.where(diff >= 0, diff, 0.1 * diff)


def LLM_Criterion(false_logits, true_logits):
    logits = torch.stack([true_logits, false_logits], dim=1)  # shape: [batch, 2]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # true is index 0
    #print(labels)
    return F.cross_entropy(logits, labels)