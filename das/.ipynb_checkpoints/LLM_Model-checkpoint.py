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
            # else:
            #     print(f"Test {dp} incorrect: {true_logits} > {false_logits}")
            #     print(f"Prediction: {outputs[0].argmax()}")
            #     print(f"True: {y_test[dp]}")
            total += 1
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}", flush=True)
    return accuracy


def make_model(model_name,LLM_test_data,Trained=True,device="cpu", dtype="float32",revision="main"): #Trained=0:pretrained, 1:fully randomized, 2:only randomize llm head
    if Trained==0:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,revision=revision).to(device)
    elif Trained==1:
        config = AutoConfig.from_pretrained(model_name)
        config.torch_dtype=dtype
        model = AutoModelForCausalLM.from_config(config).to(device)
    elif Trained==2: #Random Unembedding
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,revision=revision).to(device)
        model.lm_head.weight=nn.Parameter(model.lm_head.weight.clone())
        init.kaiming_uniform_(model.lm_head.weight, a=math.sqrt(5)) 
    elif Trained==3: #Random Embedding
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,revision=revision).to(device)
        model.model.embed_tokens.weight=nn.Parameter(model.model.embed_tokens.weight.clone())
        init.kaiming_uniform_(model.model.embed_tokens.weight, a=math.sqrt(5)) 
    elif Trained==4: #Random linked embedding and unembedding
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,revision=revision).to(device)
        init.kaiming_uniform_(model.model.embed_tokens.weight, a=math.sqrt(5)) 
    elif Trained==5: #Random Unembedding and Embedding with same norm as original
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,revision=revision).to(device)
        norm=torch.norm(model.model.embed_tokens.weight)
        init.kaiming_uniform_(model.model.embed_tokens.weight, a=math.sqrt(5)) 
        model.model.embed_tokens.weight=nn.Parameter(model.model.embed_tokens.weight/torch.norm(model.model.embed_tokens.weight)*norm)
        norm=torch.norm(model.lm_head.weight)
        init.kaiming_uniform_(model.lm_head.weight, a=math.sqrt(5)) 
        model.lm_head.weight=nn.Parameter(model.lm_head.weight/torch.norm(model.lm_head.weight)*norm)
    else:
        Exception("Unknown initialization")
    accuracy=test_model(model,LLM_test_data,device=device)
    return model,accuracy


def LLM_Criterion_Diff(false_logits, true_logits, *args, **kwargs):
    diff = false_logits - true_logits
    #score = torch.exp(diff)
    return diff.mean() # torch.where(diff >= 0, diff, 0.1 * diff)


def LLM_Criterion_targetCE(false_logits, true_logits, *args, **kwargs):
    """
    Compute the CE on only the distribution defined by the false and true logits (ignoring the rest of the distribution)
    """
    logits = torch.stack([true_logits, false_logits], dim=1)  # shape: [batch, 2]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # true is index 0
    return F.cross_entropy(logits, labels)

def LLM_Criterion_CE(false_logits, true_logits, logits, true_idx, *args, **kwargs):
    """
    Compute the CE on the entire distribution
    """
    return F.cross_entropy(logits, true_idx)