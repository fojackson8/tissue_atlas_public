import pandas as pd
import numpy as np
import os
import re
import collections
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import optimize
import pickle
# import plotly.express as px
import datetime
from scipy import stats
# from scipy.stats import zscore
import time



from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

import wandb

sys.path.append('/well/ludwig/users/dyp502/code/tissue_atlas_code_2023/')
from tissue_atlas_v3_functions import merge_filter_new_tissue_samples

sys.path.append('/well/ludwig/users/dyp502/code/taps_tissue_atlas_Nov2022/')
from taps_tissue_atlas_functions import make_tissue_average_df
from taps_tissue_atlas_functions import plot_region, plot_tissue_sample_atlas, visualise_bed



with open('/well/ludwig/users/dyp502/tissue_atlas_v3/metadata/Atlas_V3_Metadata_New.pickle', 'rb') as f:
    project_metadata = pickle.load(f)



def save_config(config_dict):

    exp_no = config_dict['exp_no']
    config_save_dir = config_dict['config_save_dir']
    config_file_name = f"config_predict_ge_{exp_no}.pkl"
    # Check if a config with the same name already exists to avoid overwriting
    config_file_path = os.path.join(config_save_dir, config_file_name)
    if not os.path.exists(config_file_path):
        with open(config_file_path, 'wb') as config_file:
            pickle.dump(experiment_config, config_file)
        print(f"Configuration saved as {config_file_path}")
    else:
        print(f"Configuration {config_file_name} already exists.")



# Load all the variables into memory    
project_dir = project_metadata['project_dir']
atlas_dir = project_metadata['atlas_dir'] 
metadata_dir = project_metadata['metadata_dir']
gene_exp_dir = project_metadata['gene_exp_dir']
intersections_dir = project_metadata['intersections_dir']
id_tissue_map = project_metadata["id_tissue_map"]
tissue_id_map = project_metadata["tissue_id_map"]
blood_cell_types = project_metadata["blood_cell_types"]
somatic_tissue_types = project_metadata["somatic_tissue_types"]
healthy_tissue_types = project_metadata["healthy_tissue_types"]
cancer_tissue_types = project_metadata["cancer_tissue_types"]
diseased_tissue_types = project_metadata["diseased_tissue_types"]
tissues_with_tumour = project_metadata["tissues_with_tumour"]
tissue_order = project_metadata["tissue_order"]
genomic_features = project_metadata["genomic_features"]
sample_outliers = project_metadata["sample_outliers"]
healthy_high_cov_samples = project_metadata["healthy_high_cov_samples"]
genomic_feature_colnames = project_metadata["genomic_feature_colnames"]
genomic_regions_dir = project_metadata['genomic_regions_dir']




# Contains coordinates of start and end of each data format. Start and End are INCLUSIVE
data_formats_indices = {}
# data_formats_indices['info'] = "Contains coordinates of start and end of each data format. Start and End are INCLUSIVE."
data_formats_indices['gene_body'] = [21,40]
data_formats_indices['5kb'] = [16,45]
data_formats_indices['10kb'] = [11,50]
data_formats_indices['20kb'] = [1,60]






experiment_config = {
    "save_dir": "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/data/",
    "results_save_dir": "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/results/",
    "model_save_dir": "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/results/models/",
    "config_save_dir": "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/configs/",
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.1,
    "remove_brain": True,
    "use_normalised_TPM": False,
#     "model_type": "ConvNet",
    "patience": 5,
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(levelname)s - %(message)s",
    "use_wandb": True,
#     "wandb_project": "predict_ge.DL_1kb.test9",
    "wandb_entity": "fojackson",
    "remove_zeros": True,
    "metrics": ["MAE", "R2"],
    "random_seed": 42
}


# config_save_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/configs/"









##   ------------------------------    Experiment 1       ------------------------------------------------------------------------

exp_no = 1
model_type = "ConvNet"
data_format = "20kb"
impute=True

data_load_name = f"epigenetic_rna_seq_merged.{data_format}.impute_{impute}."
model_name = f"1kb.{model_type}.nested_cv.exp{exp_no}.{data_load_name}"
wandb_project = f"predict_ge.{model_name}"

experiment_config['model_type'] = model_type
experiment_config['data_format'] = data_format
experiment_config['exp_no'] = exp_no
experiment_config['impute'] = impute
experiment_config['data_load_name'] = data_load_name
experiment_config['model_name'] = model_name
experiment_config['wandb_project'] = wandb_project



save_config(experiment_config)

