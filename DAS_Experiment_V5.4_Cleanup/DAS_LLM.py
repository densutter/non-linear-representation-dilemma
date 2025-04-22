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
                transformed_values = self.Transformation_Class.phi(output[:,-1])[:, col_indices]
                
                # Assign these values to the corresponding positions in source_activations
                self.source_activations[:, col_indices] = transformed_values
            elif self.mode_info[0] == "intervene":
                output_f=output.clone()#.detach().requires_grad_()
                result_tensor = self.Transformation_Class.phi(output[:,-1])
                result_tensor = torch.where(self.mode_info[1], self.source_activations, result_tensor)
                output_f[:,-1]=self.Transformation_Class.phi_inv(result_tensor)
                if twotuple:
                    return (output_f,None)
                else:
                    return (output_f,)
        
        return layer.register_forward_hook(hook_fn)

    
    def process_Batch(self,mode,data,ac_batch,total_correct,total_samples): 
        true_logits=[]
        false_logits=[]
        for ac_dp in ac_batch:
            #Prepare Source Activations
            self.source_activations = torch.zeros(1, self.Hidden_Layer_Size).to(self.Device)
            #print(data[ac_dp])
            for ac_source_pos in range(len(data[ac_dp]["sources"])):
                #add info needed for optimization... no sense in running inputs who are not used:
                self.mode_info=["source",ac_source_pos]
                ac_source=data[ac_dp]["sources"][ac_source_pos]
                if mode==1:
                    self.Model(**ac_source.to(self.Device))
                else:
                    with torch.no_grad():
                        self.Model(**ac_source.to(self.Device))

            #Intervention
            ac_base=data[ac_dp]["base"]
            intervention_bools=torch.zeros((1, self.Hidden_Layer_Size), dtype=torch.bool)
            intervention_bools[0, self.Variable_Dimensions[0]] = True
            intervention_bools=intervention_bools.to(self.Device)
            self.mode_info=["intervene",intervention_bools]
            if mode==1:
                outputs=self.Model(**ac_base.to(self.Device))["logits"][:,-1]
            else:
                with torch.no_grad():
                    outputs=self.Model(**ac_base.to(self.Device))["logits"][:,-1]
            
            labels=data[ac_dp]["label"]
            true_logits.append(outputs[0][labels[0]])
            false_logits.append(outputs[0][labels[1]])
        
        true_logits=torch.stack(true_logits)
        false_logits=torch.stack(false_logits)
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