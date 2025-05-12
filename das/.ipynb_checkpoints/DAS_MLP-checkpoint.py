import torch
import random
import copy
import torch.nn as nn
import torch.optim as optim


from .DAS import Distributed_Alignment_Search


class Distributed_Alignment_Search_MLP(Distributed_Alignment_Search):
    
    
    def Prepare_Dataset(self,Raw_Dataset): 
        Clean_Dataset={}
        Clean_Dataset["base"]=[]
        Clean_Dataset["sources"]=[]
        for _ in range(self.Num_Varibales):
            Clean_Dataset["sources"].append([])
        Clean_Dataset["label"]=[]
        Clean_Dataset["intervention"]=[]

        #Actually adapted to the dataset I am using. Needs to be changed later:
        for ac_DP_pos in range(len(Raw_Dataset)):
            #base input
            ac_DP=Raw_Dataset[ac_DP_pos]
            Clean_Dataset["base"].append(ac_DP["base"])

            #expected label after intervention
            Clean_Dataset["label"].append(ac_DP["label"])

            #source inputs
            for var_dim in range(ac_DP['sources'].shape[0]):
                Clean_Dataset["sources"][var_dim].append(ac_DP['sources'][var_dim])

            #intervention data
            Clean_Dataset["intervention"].append([])
            for Hidden_Dimension in range(self.Hidden_Layer_Size):
                dim_found=False
                for ac_dimensions_pos,ac_dimensions in enumerate(self.Variable_Dimensions):
                    if Hidden_Dimension in ac_dimensions:
                        dim_found=True
                        if ac_DP["intervention"][ac_dimensions_pos]:
                            Clean_Dataset["intervention"][-1].append(True)
                        else:
                            Clean_Dataset["intervention"][-1].append(False)
                        break
                if not dim_found:
                    Clean_Dataset["intervention"][-1].append(False)
            

        sample_number=len(Clean_Dataset["label"])
        Clean_Dataset["base"]=torch.stack(Clean_Dataset["base"]).to(self.Device)
        Clean_Dataset["label"]=torch.stack(Clean_Dataset["label"]).to(self.Device)
        for i in range(len(Clean_Dataset["sources"])):
            Clean_Dataset["sources"][i]=torch.stack(Clean_Dataset["sources"][i]).to(self.Device)
        Clean_Dataset["intervention"]=torch.tensor(Clean_Dataset["intervention"]).to(self.Device)
        return Clean_Dataset,sample_number


    def register_intervention_hook(self, layer):
        def hook_fn(module, input, output):
            if self.mode_info[0] == "source":
                # Extract row indices (list)
                row_indices = self.mode_info[2]
                
                # Extract column indices (list)
                col_indices = self.Variable_Dimensions[self.mode_info[1]]
                
                # Get values from the transformation tensor
                transformed_values = self.Transformation_Class.phi(output)[:, col_indices]
                
                # Assign these values to the corresponding positions in source_activations
                self.source_activations[torch.tensor(row_indices).unsqueeze(1), torch.tensor(col_indices).unsqueeze(0)] = transformed_values
                
            elif self.mode_info[0] == "intervene":
                result_tensor = self.Transformation_Class.phi(output)
                
                result_tensor = torch.where(self.mode_info[1], self.source_activations, result_tensor)
                
                return self.Transformation_Class.phi_inv(result_tensor)
        
        return layer.register_forward_hook(hook_fn)
    def process_Batch(self,mode,data,ac_batch,total_correct,total_samples): 
        #Prepare Source Activations
        self.source_activations = torch.zeros(len(ac_batch), self.Hidden_Layer_Size).to(self.Device)
        
        for ac_source_pos in range(len(data["sources"])):
            #add info needed for optimization... no sense in running inputs who are not used:
            used_source_indices=self.extract_sources_to_run(ac_source_pos,ac_batch,data)
            if len(used_source_indices)>0:
                self.mode_info=["source",ac_source_pos,used_source_indices]
                ac_source=data["sources"][ac_source_pos][ac_batch][used_source_indices]
                if mode==1:
                    self.Model(ac_source)
                else:
                    with torch.no_grad():
                        self.Model(ac_source)

        #Intervention
        ac_base=data["base"][ac_batch]
        intervention_bools=data["intervention"][ac_batch]
        self.mode_info=["intervene",intervention_bools]
        if mode==1:
            outputs=self.Model(ac_base)
        else:
            with torch.no_grad():
                outputs=self.Model(ac_base)
        labels=data["label"][ac_batch]

        predictions = torch.argmax(outputs, dim=1) 
        correct = (predictions.squeeze() == labels.squeeze()).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        loss = self.Transformation_Class.criterion(outputs, labels.squeeze().long())

        return loss,total_correct,total_samples,{}
        

    def extract_sources_to_run(self,which_variable,batch_indices,data):
        used_source_indices=[]
        for ip, i in enumerate(batch_indices):
            is_used=data["intervention"][i][self.Variable_Dimensions[which_variable][0]]
            if is_used:
                used_source_indices.append(ip)
        return used_source_indices