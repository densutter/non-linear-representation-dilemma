import torch
import random
import copy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

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

    def _run_validation(self, batch_size, verbose=False):
        """Runs a validation pass on the Eval_Dataset."""
        # --- Save Original States ---
        original_model_training_state = self.Model.training
        original_phi_training_state = self.Transformation_Class.phi.training
        original_model_grad_states = {name: p.requires_grad for name, p in self.Model.named_parameters()}
        original_phi_grad_states = {name: p.requires_grad for name, p in self.Transformation_Class.phi.named_parameters()}

        # --- Set to Evaluation Mode ---
        self.Model.eval()
        for param in self.Model.parameters():
            param.requires_grad = False
        self.Transformation_Class.phi.eval()
        for param in self.Transformation_Class.phi.parameters():
            param.requires_grad = False

        # --- Prepare Validation Data ---
        data = self.Eval_Dataset
        Sample_Indices = list(range(self.Eval_Samples_Number))
        Val_Batches = self.chunk_list(Sample_Indices, batch_size)
        
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        iterator = Val_Batches
        if verbose:
            print("Validation:")
            iterator = tqdm(Val_Batches)

        # --- Validation Loop ---
        for ac_batch in iterator:
            # process_Batch expects mode, data, ac_batch, total_correct, total_samples
            # It returns loss (ignored here), updated total_correct, updated total_samples
            loss, total_correct, total_samples = self.process_Batch(mode=3, data=data, ac_batch=ac_batch, 
                                                                 total_correct=total_correct, total_samples=total_samples)
            total_loss += loss.item()
            if verbose and hasattr(iterator, 'set_description'):
                accuracy = total_correct / total_samples if total_samples > 0 else 0
                iterator.set_description(f"Validation Accuracy: {accuracy:.4f}")

        accuracy = total_correct / total_samples if total_samples > 0 else 0

        # --- Restore Original States ---
        self.Model.train(original_model_training_state)
        for name, param in self.Model.named_parameters():
            param.requires_grad = original_model_grad_states.get(name, False) # Restore original grad state
        self.Transformation_Class.phi.train(original_phi_training_state)
        for name, param in self.Transformation_Class.phi.named_parameters():
             param.requires_grad = original_phi_grad_states.get(name, False) # Restore original grad state

        return accuracy, total_loss

    def train_test(self,
                   batch_size,
                   epochs=1,
                   mode=1,
                   early_stopping_threshold=3,
                   early_stopping_improve_threshold=0.001,
                   TrainModel=False,
                   verbose=False,
                   val_every_n_steps=None): # 1=train, 2=test, 3=eval

        # --- Initial State Setup based on mode and TrainModel ---
        initial_model_train_state = self.Model.training # Store original state to restore at the end
        initial_phi_train_state = self.Transformation_Class.phi.training # Store original phi state

        if mode == 1: # Training Mode
            if TrainModel:
                self.Model.train()
                for param in self.Model.parameters():
                    param.requires_grad = True
            else:
                self.Model.eval()
                for param in self.Model.parameters():
                    param.requires_grad = False
            
            self.Transformation_Class.phi.train()
            for param in self.Transformation_Class.phi.parameters():
                param.requires_grad = True
        
        elif mode == 2: # Testing Mode
             self.Model.eval()
             for param in self.Model.parameters():
                 param.requires_grad = False
             self.Transformation_Class.phi.eval()
             for param in self.Transformation_Class.phi.parameters():
                 param.requires_grad = False

        elif mode == 3: # Evaluation Mode (Now delegates entirely)
            return self._run_validation(batch_size, verbose=verbose)

        # --- Prepare Data based on mode ---
        if mode == 1:
            data = self.Train_Dataset
            Sample_Indices = list(range(self.Train_Sample_Number))
        elif mode == 2:
            data = self.Test_Dataset
            Sample_Indices = list(range(self.Test_Samples_Number))
        # mode == 3 is handled above

        # --- Training Loop Variables ---
        best_accuracy = -1
        best_accuracy_corr = -1
        steps_without_improvement = 0
        Best_Phi = None
        step = 0
        early_stop_triggered = False

        # --- Epoch Loop ---
        for epoch in range(epochs):
            total_correct = 0 # Correct count for the *training* epoch/batch
            total_samples = 0 # Sample count for the *training* epoch/batch
            total_loss = 0    # Loss for the *training* epoch

            random.shuffle(Sample_Indices)
            Data_Batches = self.chunk_list(Sample_Indices, batch_size)

            iterator = Data_Batches
            if verbose:
                if mode == 1: print(f"Epoch {epoch+1}/{epochs} Training:")
                elif mode == 2: print("Testing:")
                iterator = tqdm(iterator)

            # --- Batch Loop ---
            for ac_batch in iterator:
                step += 1

                if mode == 1: # Training Step
                    self.Transformation_Class.optimizer.zero_grad()
                    
                    loss, batch_correct, batch_samples = self.process_Batch(mode, data, ac_batch, 0, 0) # Get batch-specific counts
                    total_correct += batch_correct # Accumulate for epoch average if needed later
                    total_samples += batch_samples
                    
                    accuracy = batch_correct / batch_samples if batch_samples > 0 else 0
                    if verbose and hasattr(iterator, 'set_description'):
                         iterator.set_description(f"Loss: {loss.item():.4f}, Batch Acc: {accuracy:.4f}")

                    loss.backward()
                    self.Transformation_Class.optimizer.step()
                    total_loss += loss.item()

                    wandb.log({
                        "train/loss": loss.item(),
                        "train/accuracy": accuracy, # Log batch accuracy
                        "lr": self.Transformation_Class.scheduler.get_last_lr()[-1],
                        "epoch": epoch,
                    }, step=step)

                    # --- Intra-epoch Validation ---
                    if val_every_n_steps is not None and step % val_every_n_steps == 0:
                        eval_acc, eval_loss = self._run_validation(batch_size, verbose=False) # Use the new method
                        
                        # --- Early Stopping Check (Intra-Epoch) ---
                        if eval_acc > best_accuracy_corr:
                            best_accuracy_corr = eval_acc
                            Best_Phi = copy.deepcopy(self.Transformation_Class.phi)
                            if eval_acc - early_stopping_improve_threshold > best_accuracy:
                                best_accuracy = eval_acc
                                steps_without_improvement = 0
                            else:
                                steps_without_improvement += 1
                        else:
                            steps_without_improvement += 1
                        
                        wandb.log({
                            "val/accuracy": eval_acc,
                            "val/best_accuracy": best_accuracy_corr,
                            "val/loss": eval_loss,
                            "val/steps_without_improvement": steps_without_improvement
                        }, step=step) 

                        if verbose:
                             print(f"\nStep {step} Validation: Acc={eval_acc:.4f}, BestAcc={best_accuracy_corr:.4f}, StepsWithoutImprovement={steps_without_improvement}")

                        if steps_without_improvement >= early_stopping_threshold:
                            print(f"Early stopping triggered at step {step}.")
                            early_stop_triggered = True
                            break # Exit batch loop

                elif mode == 2: # Testing Step
                     _, batch_correct, batch_samples = self.process_Batch(mode, data, ac_batch, 0, 0)
                     total_correct += batch_correct
                     total_samples += batch_samples
                     if verbose and hasattr(iterator, 'set_description'):
                         accuracy = total_correct / total_samples if total_samples > 0 else 0
                         iterator.set_description(f"Test Accuracy: {accuracy:.4f}")
            
            # --- End of Batch Loop ---
            if early_stop_triggered and mode == 1:
                 break # Exit epoch loop

            # --- End of Epoch Processing (Only for Training Mode) ---
            if mode == 1:
                avg_epoch_loss = total_loss / len(Data_Batches) if len(Data_Batches) > 0 else 0
                current_lr = self.Transformation_Class.scheduler.get_last_lr()[-1]
                self.Transformation_Class.scheduler.step(avg_epoch_loss) # Step LR based on average epoch training loss

                final_eval_acc_report = best_accuracy_corr # Default report value

                # --- End-of-Epoch Validation ---
                if val_every_n_steps is None: 
                    eval_acc, eval_loss = self._run_validation(batch_size, verbose=verbose) # Use the new method
                    final_eval_acc_report = eval_acc # Report the actual end-of-epoch accuracy

                    # --- Early Stopping Check (End-of-Epoch) ---
                    if eval_acc > best_accuracy_corr:
                        best_accuracy_corr = eval_acc
                        Best_Phi = copy.deepcopy(self.Transformation_Class.phi)
                        if eval_acc - early_stopping_improve_threshold > best_accuracy:
                            best_accuracy = eval_acc
                            steps_without_improvement = 0
                        else:
                            steps_without_improvement += 1
                    else:
                        steps_without_improvement += 1

                    wandb.log({
                        "epoch": epoch + 1, # Log against epoch number
                        "val/accuracy": eval_acc,
                        "val/best_accuracy": best_accuracy_corr,
                        "val/learning_rate": current_lr, 
                        "val/steps_without_improvement": steps_without_improvement
                    }, step=step) # Log against the global step count at epoch end

                    if steps_without_improvement >= early_stopping_threshold:
                        print("Early stopping triggered at end of epoch.")
                        early_stop_triggered = True
                
                print(f"Epoch {epoch+1}, Avg Loss: {avg_epoch_loss:.4f}, ",
                      f"Steps w/o Improvement: {steps_without_improvement}, ",
                      f"Eval Acc (End of Epoch): {final_eval_acc_report:.4f}" if val_every_n_steps is None else f"Best Eval Acc So Far: {best_accuracy_corr:.4f}",
                      f", LR: {current_lr}", flush=True)

                if early_stop_triggered:
                    break # Exit epoch loop

        # --- After Epoch Loop ---
        if mode == 1: # Training finished
            if Best_Phi is not None:
                print(f"Loading best phi with accuracy: {best_accuracy_corr:.4f}")
                Best_Phi.to(self.Device)
                Best_Phi_inverse = Best_Phi.inverse 
                final_lr = self.Transformation_Class.scheduler.get_last_lr()[-1] if hasattr(self.Transformation_Class.scheduler, 'get_last_lr') else self.Transformation_Class.optimizer.param_groups[0]['lr']
                optimizer = optim.Adam(Best_Phi.parameters(), lr=final_lr) 
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10) 
                criterion = self.Transformation_Class.criterion
                self.Transformation_Class = phi_class(Best_Phi, Best_Phi_inverse, criterion, optimizer, scheduler)
            else:
                 print("Warning: No Best_Phi found. Keeping the current phi.")
            
            # Set final states after training: phi eval, model to its original state
            self.Transformation_Class.phi.eval()
            for param in self.Transformation_Class.phi.parameters():
                param.requires_grad = False
            
            self.Model.train(initial_model_train_state) # Restore original model state
            # Restore model grad state consistent with its training state and TrainModel flag
            model_final_grad_state = (initial_model_train_state and TrainModel) 
            for param in self.Model.parameters(): 
                 param.requires_grad = model_final_grad_state

            return best_accuracy_corr # Return best validation accuracy

        elif mode == 2: # Testing finished
            # Ensure models are in eval mode
            self.Model.eval()
            for param in self.Model.parameters(): param.requires_grad = False
            self.Transformation_Class.phi.eval()
            for param in self.Transformation_Class.phi.parameters(): param.requires_grad = False

            accuracy = total_correct / total_samples if total_samples > 0 else 0
            wandb.log({"final_test_accuracy": accuracy}, step=step) 
            return accuracy
                

    def chunk_list(self, input_list, batch_size):
        return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    
    def Cleanup(self):
        self.Hook.remove()