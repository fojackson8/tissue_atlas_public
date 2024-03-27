## GE predict README

Designing new structure for the gene expression prediction project. 

- Want a modular architecture where each script does exactly one thing. 

- Need to track experiments using config files - config files should completely specify each experiment.

 Will store a dictionary of config files, one for each experiment.



 # run.py

 - Takes exp_no as ARGUMENT, loads the corresponding config file for that experiment
 
 - Instantiates the correct model (specified in config), imported from `model.py`, then runs functions from `train.py` to run nested CV


 # model.py

 - Contains functions for all models , from simple DNN to ConvNet and Hybrid models


 # train.py

 - Functions to train and evaluate models