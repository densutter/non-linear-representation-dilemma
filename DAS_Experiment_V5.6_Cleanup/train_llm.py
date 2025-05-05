# %%
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
import os
from huggingface_hub import hf_hub_download
import wandb
from transformers import AutoTokenizer
import argparse # Import argparse
from Helper_Functions import set_seed
from Dataset_Generation import Generate_LLM_Eval_Intervention_Data
from LLM_Model import (make_model,
                       LLM_Criterion)
from RevNet import RevNet
from Rotation_Model import Rotation
from DAS import phi_class
from DAS_LLM import Distributed_Alignment_Search_LLM

# %%

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run Distributed Alignment Search Experiment for LLMs")

    # Model Config
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B", help="Name of the HuggingFace model to use.")
    parser.add_argument('--model_init_mode', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Model initialization mode: 0:pretrained, 1:fully randomized, 2:only randomize llm head, 3: only randomize embedding, 4: randomize linked embedding and lm head.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run the model on ('cuda:0', 'cpu', etc.).")

    # Transformation Config
    parser.add_argument('--transformation_type', type=str, default="RevNet", choices=["RevNet", "Rotation"], help="Type of transformation function.")
    parser.add_argument('--in_features', type=int, default=2048, help="Input features for the transformation function (hidden layer size).")
    # RevNet specific
    parser.add_argument('--revnet_blocks', type=int, default=10, help="Number of blocks for RevNet transformation (if used).")
    parser.add_argument('--revnet_hidden_size', type=int, default=16, help="Hidden size for RevNet transformation (if used).")

    # Training Hyperparameters
    parser.add_argument('--max_epochs', type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument('--early_stopping_epochs', type=int, default=10, help="Patience for early stopping.")
    parser.add_argument('--early_stopping_improve_threshold', type=float, default=0.001, help="Minimum improvement threshold for early stopping.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for the transformation optimizer.")
    parser.add_argument('--lr_patience', type=int, default=10, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument('--train_batch_size', type=int, default=256, help="Batch size for training.")
    parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size for testing/evaluation.")
    parser.add_argument('--seed', type=int, default=4287, help="Random seed for reproducibility.")

    # Dataset Config
    parser.add_argument('--dataset_repo_id', type=str, default="fahamu/ioi", help="HuggingFace dataset repository ID.")
    parser.add_argument('--dataset_filename', type=str, default="mecha_ioi_200k.parquet", help="Filename of the dataset within the repository.")
    parser.add_argument('--llm_test_samples', type=int, default=1600, help="Number of samples for initial LLM testing.")
    parser.add_argument('--intervention_train_size', type=int, default=50000, help="Number of samples for intervention training.")
    parser.add_argument('--intervention_eval_size', type=int, default=1600, help="Number of samples for intervention evaluation.")
    parser.add_argument('--intervention_test_size', type=int, default=1600, help="Number of samples for intervention testing.")

    # Experiment Specific Config
    parser.add_argument('--layer_index', type=int, default=9, help="Index of the model layer to apply the intervention.")
    parser.add_argument('--intervention_dim_proportion', type=float, default=0.5, help="Proportion of dimensions to intervene on (e.g., 0.5 for the first half).")

    # Wandb Config
    parser.add_argument('--wandb_project', type=str, default="CausalAbstractionLLM", help="Wandb project name.")
    parser.add_argument('--wandb_entity', type=str, default="jkminder", help="Wandb entity (username or team name).")
    parser.add_argument('--wandb_run_name_prefix', type=str, default="InitMode", help="Prefix for the wandb run name.")

    return parser.parse_args()

# Main execution function
def main(args):
    # Set Seed right at the beginning
    set_seed(args.seed)

    # Use args for configuration
    model_config = {"model"   : args.model_name,
                    "Trained" : args.model_init_mode}

    DEVICE = args.device

    transformation_config = {"type"        : args.transformation_type,
                             "in_features" : args.in_features}
    if args.transformation_type == "RevNet":
        transformation_config["number_blocks"] = args.revnet_blocks
        transformation_config["hidden_size"] = args.revnet_hidden_size

    # %%
    # Define dataset details
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading '{args.dataset_repo_id}' file '{args.dataset_filename}'...")
    # Seed dataset generation separately if needed, or rely on global seed
    set_seed(0) # Keep original seeding for dataset generation? Or use args.seed? Using 0 for now as in original.
    dataset_path = hf_hub_download(
        repo_id=args.dataset_repo_id,
        filename=args.dataset_filename,
        repo_type="dataset"
    )
    LLM_test_data,DAS_Train,DAS_Eval,DAS_Test=Generate_LLM_Eval_Intervention_Data(filename=dataset_path,
                                                                                  tokenizer=tokenizer,
                                                                                  LLM_test_samples=args.llm_test_samples,
                                                                                  Intervention_train_size=args.intervention_train_size,
                                                                                  Intervention_eval_size=args.intervention_eval_size,
                                                                                  Intervention_test_size=args.intervention_test_size)
    # %%
    run_name = f"{args.wandb_run_name_prefix}{model_config['Trained']}_{transformation_config['type']}_Layer{args.layer_index}_DimProp{args.intervention_dim_proportion}"
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args), # Log all args
        entity=args.wandb_entity,
        name=run_name
    )

    results=[]
    results.append({})
    # Set seed again before model creation and training using the main seed
    set_seed(args.seed)
    model,accuracy=make_model(model_config["model"],LLM_test_data,model_config["Trained"],device=DEVICE)
    # Log initial model accuracy
    wandb.log({"initial_model_accuracy": accuracy})
    # %%
    Layers=[]
    # Select layer based on args
    try:
        selected_layer = model.model.layers[args.layer_index]
        layer_name = f"Layer{args.layer_index}"
        Layers.append((layer_name, selected_layer))
    except IndexError:
        print(f"Error: Layer index {args.layer_index} is out of bounds for the model.")
        wandb.finish()
        return

    inter_dims=[]
    # Define intervention dimensions based on args
    num_inter_dims = int(args.in_features * args.intervention_dim_proportion)
    inter_dims.append([list(range(0, num_inter_dims))])

    results[-1]["accuracy"]=accuracy
    for LayerName, Layer in Layers: # Loop will run once with the selected layer
        results[-1][LayerName]={}
        for inter_dim in inter_dims: # Loop will run once with the selected dimensions
            print(f"Running DAS on {LayerName}: Intervening on {len(inter_dim[0])} dimensions ({args.intervention_dim_proportion*100}%)", flush=True)
            wandb.config.update({"intervention_dimensions": len(inter_dim[0]), "layer_name": LayerName}, allow_val_change=True) # Log actual dims used

            #Initialize transformation function
            if transformation_config["type"]=="Rotation":
                p = Rotation(transformation_config["in_features"])
            elif transformation_config["type"]=="RevNet":
                p = RevNet(number_blocks =  transformation_config["number_blocks"],
                            in_features   =  transformation_config["in_features"],
                            hidden_size   =  transformation_config["hidden_size"]
                            )
            else:
                # This case should ideally be caught by argparse choices, but good practice to keep
                raise ValueError("Unknown transformation function type specified")
            p.to(DEVICE)
            p_inverse = p.inverse
            optimizer = optim.Adam(p.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.lr_patience)

            phi=phi_class(p,p_inverse,LLM_Criterion,optimizer,scheduler)


            DAS_Experiment=Distributed_Alignment_Search_LLM(Model=model,
                                                                Model_Layer=Layer,
                                                                Train_Data_Raw=DAS_Train,
                                                                Test_Data_Raw=DAS_Test,
                                                                Eval_Data_Raw=DAS_Eval,
                                                                Hidden_Layer_Size=transformation_config["in_features"],
                                                                Variable_Dimensions=inter_dim,
                                                                Transformation_Class=phi,
                                                                Device=DEVICE,
                                                                tokenizer=tokenizer)

            DAS_Experiment.train_test(batch_size=args.train_batch_size, # Use train batch size
                                        epochs=args.max_epochs,
                                        mode=1,
                                        early_stopping_threshold=args.early_stopping_epochs,
                                        early_stopping_improve_threshold=args.early_stopping_improve_threshold,
                                        verbose=True) #Train

            accuracy=DAS_Experiment.train_test(batch_size=args.test_batch_size, # Use test batch size
                                                mode=2,
                                                verbose=True) #Test

            results[-1][LayerName][str(inter_dim)]=accuracy
            DAS_Experiment.Cleanup()
            DAS_Experiment=None
            # Save results incrementally
            results_filename = f"{run_name}_results.json"
            with open(results_filename, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {results_filename}")
            # Optionally, log results file to wandb
            # wandb.save(results_filename)


    # Finish wandb run
    wandb.finish()
    print("Experiment finished.")
# %%

# Entry point for the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
