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




def load_config(exp_no):

    if not config_dir:
        config_dir = "/well/ludwig/users/dyp502/tissue_atlas_v3/predict_ge/1kb_model/finalise_model/configs/"

    config_file = f"config_predict_ge_{exp_no}.pkl"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found for experiment number {exp_no}.")
    
    with open(config_dir + config_file, "rb") as f:
        config = pickle.load(f)
    
    return config
