import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import wandb
import sys


# from train import train_and_evaluate
sys.path.append("/well/ludwig/users/dyp502/code/tissue_atlas_code_2023/gene_expression/ge_predict_final/")
import model
import train


def main(config):
    data_tensor_filtered = np.load(config["save_dir"] + config["data_load_name"] + "data_tensor_filtered.npy")
    rna_seq_values = np.load(config["save_dir"] + config["data_load_name"] + "rna_seq_values.npy")
    rna_seq_values_df = pd.read_csv(config["save_dir"] + config["data_load_name"] + "gene_tissue_combinations_rnaseq.csv", sep='\t')
    
    if config["remove_brain"]:
        rna_seq_values = rna_seq_values[rna_seq_values_df['tissue_x'] != 'Brain']
        data_tensor_filtered = data_tensor_filtered[rna_seq_values_df['tissue_x'] != 'Brain']
        rna_seq_values_df = rna_seq_values_df.loc[rna_seq_values_df['tissue_x'] != 'Brain']
        rna_seq_values_df = rna_seq_values_df.reset_index()
    
    if config["use_wandb"]:
        wandb.login()
        wandb.init(project=config["wandb_project"], config=config, entity=config["wandb_entity"])
    
    model = get_model(config["model_type"], config["n_epigenetic_cols"], config["n_1kb_bins"])
    
    results_df, all_tissue_predictions_df = train_and_evaluate(config, data_tensor_filtered, rna_seq_values, rna_seq_values_df, model)
    
    
    save_config(config, config["config_save_dir"])

if __name__ == "__main__":
   
    
    main(config)