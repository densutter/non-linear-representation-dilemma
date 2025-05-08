import sys
import os
sys.path.append(os.getcwd())
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

print("Current working directory:", os.getcwd())
from das.Helper_Functions import set_seed
from das.Dataset_Generation import Generate_LLM_Eval_Intervention_Data
from das.LLM_Model import (make_model,
                       LLM_Criterion_targetCE,
                       LLM_Criterion_Diff,
                       LLM_Criterion_CE)
from das.RevNet import RevNet
from das.Rotation_Model import Rotation
from das.DAS import phi_class
from das.DAS_LLM import Distributed_Alignment_Search_LLM



# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run Distributed Alignment Search Experiment for LLMs")

    # Model Config
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B", help="Name of the HuggingFace model to use.")
    parser.add_argument('--model_revision', type=str, default="main", help="Revision of the model to use.")
    parser.add_argument('--model_init_mode', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Model initialization mode: 0:pretrained, 1:fully randomized, 2:only randomize llm head, 3: only randomize embedding, 4: randomize linked embedding and lm head.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run the model on ('cuda:0', 'cpu', etc.).")
    parser.add_argument('--dtype', type=str, default="bfloat16", choices=["float32", "float64", "bfloat16"], help="Data type for the model.")

    # Transformation Config
    parser.add_argument('--transformation_type', type=str, default="RevNet", choices=["RevNet", "Rotation"], help="Type of transformation function.")
    parser.add_argument('--in_features', type=int, default=2048, help="Input features for the transformation function (hidden layer size).")
    # RevNet specific
    parser.add_argument('--revnet_blocks', type=int, default=10, help="Number of blocks for RevNet transformation (if used).")
    parser.add_argument('--revnet_hidden_size', type=int, default=16, help="Hidden size for RevNet transformation (if used).")
    parser.add_argument('--revnet_depth', type=int, default=1, help="Depth for RevNet transformation (if used).")

    # Training Hyperparameters
    parser.add_argument('--max_epochs', type=int, default=1, help="Maximum number of training epochs.")
    parser.add_argument('--early_stopping_epochs', type=int, default=3, help="Patience for early stopping.")
    parser.add_argument('--early_stopping_improve_threshold', type=float, default=0.01, help="Minimum improvement threshold for early stopping.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for the transformation optimizer.")
    parser.add_argument('--lr_patience', type=int, default=10, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument('--train_batch_size', type=int, default=256, help="Batch size for training.")
    parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size for testing/evaluation.")
    parser.add_argument('--seed', type=int, default=4287, help="Random seed for reproducibility.")
    parser.add_argument('--val_every_n_steps', type=int, default=100, help="Number of steps between validation checks.")
    parser.add_argument('--diff_loss', action="store_true", help="Use difference loss instead of CE loss.")
    parser.add_argument('--target_ce_loss', action="store_true", help="Use target CE loss instead of CE loss.")
    # Dataset Config
    parser.add_argument('--dataset_repo_id', type=str, default="fahamu/ioi", help="HuggingFace dataset repository ID.")
    parser.add_argument('--dataset_filename', type=str, default="mecha_ioi_26m.parquet", help="Filename of the dataset within the repository.")
    parser.add_argument('--llm_test_samples', type=int, default=2**14, help="Number of samples for initial LLM testing.")
    parser.add_argument('--intervention_train_size', type=int, default=1000000, help="Number of samples for intervention training.")
    parser.add_argument('--intervention_eval_size', type=int, default=2048, help="Number of samples for intervention evaluation.")
    parser.add_argument('--intervention_test_size', type=int, default=2**14, help="Number of samples for intervention testing.")

    # Experiment Specific Config
    parser.add_argument('--layer_index', type=int, default=9, help="Index of the model layer to apply the intervention.")
    parser.add_argument('--intervention_dim_proportion', type=float, default=0.5, help="Proportion of dimensions to intervene on (e.g., 0.5 for the first half).")
    parser.add_argument('--results_path', type=str, default="results", help="Path to save the results.")

    # Wandb Config
    parser.add_argument('--wandb_project', type=str, default="CausalAbstractionLLM", help="Wandb project name.")
    parser.add_argument('--wandb_entity', type=str, default="jkminder", help="Wandb entity (username or team name).")
    parser.add_argument('--run_name', type=str, default="", help="Prefix for the wandb run name.")
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help="Number of linear learning rate warmup steps.")
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
        transformation_config["depth"] = args.revnet_depth
    
    # Define dataset details
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float64
    
    print(f"Loading datasets from {args.model_name.replace('/', '_')}")
    if os.path.exists(f"datasets/{args.model_name.replace('/', '_')}"):
        LLM_test_data = torch.load(f"datasets/{args.model_name.replace('/', '_')}/LLM_test_data.pt")[:args.llm_test_samples]
        DAS_Train = torch.load(f"datasets/{args.model_name.replace('/', '_')}/DAS_Train.pt")[:args.intervention_train_size]
        DAS_Eval = torch.load(f"datasets/{args.model_name.replace('/', '_')}/DAS_Eval.pt")[:args.intervention_eval_size]
        DAS_Test = torch.load(f"datasets/{args.model_name.replace('/', '_')}/DAS_Test.pt")[:args.intervention_test_size]
    else:
        print("Generating datasets...")
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
        
        os.makedirs(f"datasets/{args.model_name.replace('/', '_')}")
        torch.save(LLM_test_data, f"datasets/{args.model_name.replace('/', '_')}/LLM_test_data.pt")
        torch.save(DAS_Train, f"datasets/{args.model_name.replace('/', '_')}/DAS_Train.pt")
        torch.save(DAS_Eval, f"datasets/{args.model_name.replace('/', '_')}/DAS_Eval.pt")
        torch.save(DAS_Test, f"datasets/{args.model_name.replace('/', '_')}/DAS_Test.pt")

    transformation_name = ""
    if args.transformation_type == "RevNet":
        transformation_name = f"B{args.revnet_blocks}H{args.revnet_hidden_size}D{args.revnet_depth}"
    run_name = f"{args.run_name}{args.model_revision}_m{model_config['Trained']}_l{args.layer_index}_{transformation_config['type']}{transformation_name}_{args.dtype}_lr{args.lr:.1e}"
    assert not (args.diff_loss and args.target_ce_loss), "Cannot use both difference and target CE loss"
    if args.diff_loss:
        run_name += "_diff"
    if args.target_ce_loss:
        run_name += "_target_ce"
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
    model,accuracy=make_model(model_config["model"],LLM_test_data,model_config["Trained"],device=DEVICE,revision=args.model_revision)
    if "pythia" in args.model_name:
        print("Patching model.model to model.gpt_neox for pythia models")
        setattr(model, "model", model.gpt_neox)
    # Log initial model accuracy
    wandb.log({"test/model_accuracy": accuracy})
    
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
                p = RevNet( number_blocks =  transformation_config["number_blocks"],
                            in_features   =  transformation_config["in_features"],
                            hidden_size   =  transformation_config["hidden_size"],
                            depth         =  transformation_config["depth"]
                            )
            else:
                # This case should ideally be caught by argparse choices, but good practice to keep
                raise ValueError("Unknown transformation function type specified")
            p.to(DEVICE).to(dtype)
            model.to(dtype)
            p_inverse = p.inverse
            optimizer = optim.Adam(p.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.lr_patience)

            if args.diff_loss:
                phi=phi_class(p,p_inverse,LLM_Criterion_Diff,optimizer,scheduler)
            elif args.target_ce_loss:
                phi=phi_class(p,p_inverse,LLM_Criterion_targetCE,optimizer,scheduler)
            else:
                phi=phi_class(p,p_inverse,LLM_Criterion_CE,optimizer,scheduler)


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

            DAS_Experiment.train_test(
                batch_size=args.train_batch_size,
                epochs=args.max_epochs,
                mode=1,
                early_stopping_threshold=args.early_stopping_epochs,
                early_stopping_improve_threshold=args.early_stopping_improve_threshold,
                verbose=True,
                lr_warmup_steps=args.lr_warmup_steps,
                val_every_n_steps=args.val_every_n_steps
            )

            accuracy=DAS_Experiment.train_test(batch_size=args.test_batch_size, # Use test batch size
                                                mode=2,
                                                verbose=True) #Test

            results[-1][LayerName][str(inter_dim)]=accuracy
            DAS_Experiment.Cleanup()
            DAS_Experiment=None
            # Save results incrementally
            args.results_path = f"{args.results_path}/{args.model_name.replace('/', '_')}"
            os.makedirs(args.results_path, exist_ok=True)
            results_filename = f"{args.results_path}/{run_name}_results.json"
            with open(results_filename, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {results_filename}")
            # Optionally, log results file to wandb
            wandb.save(results_filename)


    # Finish wandb run
    wandb.finish()
    print("Experiment finished.")


# Entry point for the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
