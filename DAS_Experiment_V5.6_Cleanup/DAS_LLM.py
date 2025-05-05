import torch
import random
import copy
import torch.nn as nn
import torch.optim as optim

from LLM_Model import LLM_Criterion
from DAS import Distributed_Alignment_Search

class Distributed_Alignment_Search_LLM(Distributed_Alignment_Search):

    def Prepare_Dataset(self,Raw_Dataset): 
        return (Raw_Dataset,len(Raw_Dataset))
        
    def register_intervention_hook(self, layer):
        def hook_fn(module, input, output):
            twotuple=(len(output)==2)
            output=output[0].clone().detach()
            if self.mode_info[0] == "source":
                # Extract column indices (list)
                col_indices = self.Variable_Dimensions[self.mode_info[1]]
                
                # Get values from the transformation tensor
                transformed_values = self.Transformation_Class.phi(output[torch.arange(output.size(0)), [self.mode_info[3][i] for i in self.mode_info[2]]])[:, col_indices]
                # Assign these values to the corresponding positions in source_activations
                self.source_activations[torch.tensor(self.mode_info[2]).unsqueeze(1), torch.tensor(col_indices).unsqueeze(0)] = transformed_values
                
            elif self.mode_info[0] == "intervene":
                output_f=output.clone()#.detach().requires_grad_()
                result_tensor = self.Transformation_Class.phi(output[torch.arange(output.size(0)), self.mode_info[2]])
                result_tensor = torch.where(self.mode_info[1], self.source_activations, result_tensor)
                output_f[torch.arange(output.size(0)), self.mode_info[2]]=self.Transformation_Class.phi_inv(result_tensor)
                #output_f[:,-1]=self.Transformation_Class.phi_inv(result_tensor)
                if twotuple:
                    return (output_f,None)
                else:
                    return (output_f,)
        
        return layer.register_forward_hook(hook_fn)

    
    def process_Batch(self,mode,data,ac_batch,total_correct,total_samples): 
        true_logits=[]
        false_logits=[]
        self.source_activations = torch.zeros(len(ac_batch), self.Hidden_Layer_Size).to(self.Device)
        base_text=[]
        base_output_pos=[]
        source_text=[]
        source_output_pos=[]
        source_used=[]
        variable_num=len(data[0]["intervention"])
        intervention_bools=torch.zeros((len(ac_batch),self.Hidden_Layer_Size), dtype=torch.bool)
        
        for _ in range(variable_num):
            source_text.append([])
            source_output_pos.append([])
            source_used.append([])
        
        idx_true_logit=[]
        idx_false_logit=[]
        for ac_pos,ac_dp in enumerate(ac_batch):
            base_text.append(data[ac_dp]["base"])
            base_output_pos.append(len(data[ac_dp]["base"]['input_ids'])-1)
            for aS in range(variable_num):
                if data[ac_dp]["intervention"][aS]:
                    source_text[aS].append(data[ac_dp]["sources"][aS])
                    source_output_pos[aS].append(len(data[ac_dp]["sources"][aS]['input_ids'])-1)
                    source_used[aS].append(ac_pos)
                    intervention_bools[ac_pos, self.Variable_Dimensions[aS]] = True
            idx_true_logit.append(data[ac_dp]["label"][0])
            idx_false_logit.append(data[ac_dp]["label"][1])

        intervention_bools=intervention_bools.to(self.Device)
        base_text=self.tokenizer.pad(base_text, padding=True, return_tensors="pt").to(self.Device)
        for aS in range(variable_num):
            if len(source_used[aS])>0:
                source_text[aS]=self.tokenizer.pad(source_text[aS], padding=True, return_tensors="pt").to(self.Device)
            
                self.mode_info=["source",aS,source_used[aS],source_output_pos[aS]]
                if mode==1:
                    self.Model(**source_text[aS])
                else:
                    with torch.no_grad():
                        self.Model(**source_text[aS])
    
        self.mode_info=["intervene",intervention_bools,base_output_pos]
        if mode==1:
            outputs=self.Model(**base_text)["logits"]
        else:
            with torch.no_grad():
                outputs=self.Model(**base_text)["logits"]
        
        true_logits=outputs[torch.arange(outputs.size(0)), base_output_pos, idx_true_logit]
        false_logits=outputs[torch.arange(outputs.size(0)), base_output_pos, idx_false_logit]
        loss=0
        if mode==1:
            #print("t:",torch.stack([true_logits, false_logits], dim=1))
            loss = self.Transformation_Class.criterion(false_logits, true_logits)

        else:
            #print("e:",torch.stack([true_logits, false_logits], dim=1))
            total_correct += torch.sum(true_logits>false_logits).item()
            total_samples += true_logits.shape[0]

        
        return loss,total_correct,total_samples


    
    def chunk_list(self, input_list, batch_size):
        return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    
    def Cleanup(self):
        self.Hook.remove()