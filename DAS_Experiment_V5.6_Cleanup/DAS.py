import torch
import random
import copy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class phi_class:
    def __init__(self, phi,phi_inv,criterion,optimizer,scheduler):
        self.phi=phi
        self.phi_inv=phi_inv
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler

class Distributed_Alignment_Search:

    
    def __init__(self, 
                 Model,
                 Model_Layer,
                 Train_Data_Raw,
                 Test_Data_Raw,
                 Eval_Data_Raw,
                 Hidden_Layer_Size,
                 Variable_Dimensions,
                 Transformation_Class,
                 Device,
                 tokenizer=None):
        self.Model=Model
        self.Model_Layer=Model_Layer
        self.Hidden_Layer_Size=Hidden_Layer_Size
        self.Variable_Dimensions=Variable_Dimensions
        self.Num_Varibales=len(self.Variable_Dimensions)
        self.Transformation_Class=Transformation_Class
        self.Device=Device
        self.tokenizer=tokenizer

        self.mode_info=None
        self.base_activations=None
        self.source_activations=None

        
        self.Train_Dataset,self.Train_Sample_Number=self.Prepare_Dataset(Train_Data_Raw)
        self.Test_Dataset,self.Test_Samples_Number=self.Prepare_Dataset(Test_Data_Raw)
        self.Eval_Dataset,self.Eval_Samples_Number=self.Prepare_Dataset(Eval_Data_Raw)

   
        self.Hook=self.register_intervention_hook(self.Model_Layer)
        

    def Prepare_Dataset(self,Raw_Dataset):
        pass

    def register_intervention_hook(self, layer):       
        pass

    def process_Batch(self,mode,data,ac_batch,total_correct,total_samples): 
        pass

    def train_test(self,
                   batch_size,epochs=1,
                   mode=1,
                   early_stopping_threshold=3,
                   early_stopping_improve_threshold=0.001,
                   TrainModel=False,
                   verbose=False): #1=train,2=test,3=eval


        if TrainModel and mode==1:
            self.Model.train()
            for param in self.Model.parameters():
                param.requires_grad = True  # This unfreezes the weights
        else:
            self.Model.eval()
            for param in self.Model.parameters():
                param.requires_grad = False  # This freezes the weights
        
        if mode==1:
            self.Transformation_Class.phi.train()
            for param in self.Transformation_Class.phi.parameters():
                param.requires_grad = True  # This unfreezes the weights
        elif mode in [2,3]:
            self.Transformation_Class.phi.eval()
            for param in self.Transformation_Class.phi.parameters():
                param.requires_grad = False  # This freezes the weights

        if mode==1:
            data=self.Train_Dataset
            Sample_Indices=list(range(self.Train_Sample_Number))
        elif mode==2:
            data=self.Test_Dataset
            Sample_Indices=list(range(self.Test_Samples_Number))
        elif mode==3:
            data=self.Eval_Dataset
            Sample_Indices=list(range(self.Eval_Samples_Number))
        
        best_accuracy=-1
        best_accuracy_corr=-1
        steps_without_improvement=0
        Best_Phi=None
        for epoch in range(epochs):
            total_correct = 0
            total_samples = 0
            total_loss = 0
        
            #Make Batches
            random.shuffle(Sample_Indices)
            DAS_Train_Batches=self.chunk_list(Sample_Indices, batch_size)
            
            iterator=DAS_Train_Batches
            if verbose:
                if mode==1:
                    print("Training:")
                elif mode==2:
                    print("Testing:")
                elif mode==3:
                    print("Eval:")
                iterator=tqdm(iterator)
            for ac_batch in iterator:
                #print(ac_batch)

                if mode==1: 
                    self.Transformation_Class.optimizer.zero_grad()
                
                
                loss,total_correct,total_samples = self.process_Batch(mode,data,ac_batch,total_correct,total_samples)
                
                if mode==1:
                    loss.backward()
                    self.Transformation_Class.optimizer.step()
                    total_loss += loss.item()

            #Early stopping mechanism
            if mode==1:
                eval_acc=self.train_test(batch_size,epochs=1,mode=3,verbose=verbose,TrainModel=TrainModel)
                if eval_acc>best_accuracy_corr:
                    best_accuracy_corr=eval_acc
                    Best_Phi=copy.deepcopy(self.Transformation_Class.phi)
                    if eval_acc-early_stopping_improve_threshold>best_accuracy:
                        best_accuracy=eval_acc
                        steps_without_improvement=0
                    else:
                        steps_without_improvement+=1
                else:
                    steps_without_improvement+=1
                    
                total_loss=total_loss / len(DAS_Train_Batches)
                self.Transformation_Class.scheduler.step(total_loss)
                print(f"Epoch {epoch+1}, Loss: {total_loss}",
                      "steps without improvement:",steps_without_improvement,
                      "eval accuracy:",eval_acc,
                      "best eval accuracy:",best_accuracy_corr,
                      "learning rate:",self.Transformation_Class.scheduler.get_last_lr()[-1], flush=True)
                
                if steps_without_improvement>=early_stopping_threshold or epoch==epochs-1:
                    Best_Phi.to(self.Device)
                    Best_Phi_inverse=Best_Phi.inverse
                    criterion = self.Transformation_Class.criterion
                    optimizer = optim.Adam(Best_Phi.parameters(), lr=self.Transformation_Class.scheduler.get_last_lr()[-1])
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
                    self.Transformation_Class=phi_class(Best_Phi,Best_Phi_inverse,criterion,optimizer,scheduler)
                    break
                
        if mode==2 or mode==3: 
            self.Transformation_Class.phi.train()
            for param in self.Transformation_Class.phi.parameters():
                param.requires_grad = True  # This unfreezes the weights
            accuracy = total_correct / total_samples
            if TrainModel:
                self.Model.train()
                for param in self.Model.parameters():
                    param.requires_grad = True  # This unfreezes the weights
            return accuracy
                

    def chunk_list(self, input_list, batch_size):
        return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    
    def Cleanup(self):
        self.Hook.remove()