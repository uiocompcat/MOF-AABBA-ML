import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
import pandas as pd

import tools
from nets import ExampleNet
from trainer import Trainer
from dataset import CustomDataset
from plot import plot_correlation, plot_target_histogram

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import processing_data
from processing_data import list_all_aabba_features_rac_mult
from processing_data import list_all_aabba_features_rac_sum
from pathlib import Path
import os

import csv
import schedule
import time

# PARAMETERS
wandb_entity = 'nkuriakose'
seed = 40870
learning_rate = 0.01
batch_size = 32
n_epochs = 250

#model = ExampleNet(input_nodes=len(features_input.columns), hidden_nodes=hidden_nodes, output_nodes=1)


def run_job(wandbname, property, dnn):

    print('new iteratio of the running')

    wandb.require("core")
    wandb_project_name = wandbname
    wandb_run_name = f'custom_{property}_e{n_epochs}_b{batch_size}_dnn{dnn}' #'FA_AA_d8_homo_lumo_gap_delta_epoch200_b32'

    # CODE
    wandb.init(project=wandb_project_name, entity=wandb_entity)
    # set name
    wandb.run.name = wandb_run_name

    # set seed
    tools.set_global_seed(seed)

    # setup data set
    data = pd.read_csv('core8268_dd3k_H2mgpg_gcmc50k_geo_aabba_d05.csv') #low_memory=False
     
    
    
    #mof_chem_descriptor_column_prefix = ['mc', 'D_mc', 'lc', 'D_lc', 'f-lig', 'func', 'D_func']
    #mof_chem_descriptors = [column for column in data.columns if any(column.startswith(prefix) for prefix in mof_chem_descriptor_column_prefix)]
    mof_geom_descriptors = ['CellV', 'Df', 'Di', 'Dif', 'density', 'total_SA_volumetric', 'total_SA_gravimetric', 'total_POV_volumetric', 'total_POV_gravimetric']
    
    features_input = mof_geom_descriptors + list_all_aabba_features_rac_mult() + list_all_aabba_features_rac_sum()

    print('len input rac mult', len(list_all_aabba_features_rac_mult()))
    print('len input rac sum', len(list_all_aabba_features_rac_sum()))
    print('len input', len(features_input)) 


    # select the data
    sub = data[features_input]
    print('len sub', len(sub))
    
    target = data[property]
    print('len target', len(target))

    X_train, X_test, y_train, y_test = train_test_split(sub, target, test_size=0.2, train_size=0.8, random_state=2024, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=2024, shuffle=True)

    # standard scale subselection
    X_train, mean, std, outliers = processing_data.standarize_train(X_train)
    print('outliers', len(outliers), 'total', len(features_input)-len(outliers))
    X_val = processing_data.standarize_rest(X_val, mean, std, outliers)
    X_test = processing_data.standarize_rest(X_test, mean, std, outliers)

    X_train_csv = X_train.to_csv(f'xtrain_{wandb_project_name}.csv')
    y_train_csv = y_train.to_csv(f'ytrain_{wandb_project_name}.csv')
    X_val_csv = X_val.to_csv(f'xval_{wandb_project_name}.csv')
    y_val_csv = y_val.to_csv(f'yval_{wandb_project_name}.csv')
    X_test_csv = X_test.to_csv(f'xtest_{wandb_project_name}.csv')
    y_test_csv = y_test.to_csv(f'ytest_{wandb_project_name}.csv')

    print('subset', len(sub), 'train', len(X_train), X_train, 'val', len(X_val), X_val, 'test', len(X_test), X_test)

    # cast variables to torch tensors
    X_train_torch = torch.tensor(X_train.values, dtype=torch.float)
    y_train_torch = torch.tensor(y_train.values.reshape((-1, 1)), dtype=torch.float)
    dataset_train = CustomDataset(X_train_torch, y_train_torch)

    X_val_torch = torch.tensor(X_val.values, dtype=torch.float)
    y_val_torch = torch.tensor(y_val.values.reshape((-1, 1)), dtype=torch.float)
    dataset_val = CustomDataset(X_val_torch, y_val_torch)

    X_test_torch = torch.tensor(X_test.values, dtype=torch.float)
    y_test_torch = torch.tensor(y_test.values.reshape((-1, 1)), dtype=torch.float)
    dataset_test = CustomDataset(X_test_torch, y_test_torch)

    print('Using ' + str(len(sub)) + ' data points. (train=' + str(len(X_train_torch)) + ', val=' + str(len(X_val_torch)) + ', test=' + str(len(X_test_torch)) + ')')

    # set up dataloaders for Vaska's dataset

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_unshuffled = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if dnn == 0:
        hidden_nodes = 64 
    if dnn == 1:
        hidden_nodes = 128
    if dnn == 2:
        hidden_nodes = 256
    if dnn == 3:
        hidden_nodes = 512    

    # set up model
    model = ExampleNet(input_nodes=len(features_input)-len(outliers), hidden_nodes=hidden_nodes, output_nodes=1)

    # set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

    # run
    trainer = Trainer(model, optimizer, scheduler)
    print('Starting training..')

    trained_model = trainer.run(train_loader,
                                train_loader_unshuffled,
                                val_loader, test_loader,
                                n_epochs=n_epochs)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), '/home/gc_s40870_802050k_l3n1_ddcr_sum_mul_geom_depth2/trained_model.pth')

    # get training set predictions and ground truths
    train_predicted_values = trainer.predict_loader(train_loader_unshuffled)
    train_true_values = tools.get_target_list(train_loader_unshuffled)

    # get validation set predictions and ground truths
    val_predicted_values = trainer.predict_loader(val_loader)
    val_true_values = tools.get_target_list(val_loader)

    # get test set predictions and ground truths
    test_predicted_values = trainer.predict_loader(test_loader)
    test_true_values = tools.get_target_list(test_loader)

    # log predictions

    train_df = pd.DataFrame({'predicted': train_predicted_values,
    'truth': train_true_values})
    wandb.log({"train-predictions": wandb.Table(dataframe=train_df)})

    val_df = pd.DataFrame({'predicted': val_predicted_values,
    'truth': val_true_values})
    wandb.log({"val-predictions": wandb.Table(dataframe=val_df)})

    test_df = pd.DataFrame({'predicted': test_predicted_values,
    'truth': test_true_values})
    wandb.log({"test-predictions": wandb.Table(dataframe=test_df)})

    # log plots

    tmp_file_path = '/home/gc_s40870_802050k_l3n1_ddcr_sum_mul_geom_depth2/image.png'
    #tmp_file_path = '/tmp/image.png'

    plot_correlation(train_predicted_values, train_true_values, file_path=tmp_file_path)
    wandb.log({'Training set prediction correlation': wandb.Image(tmp_file_path)})

    plot_correlation(val_predicted_values, val_true_values, file_path=tmp_file_path)
    wandb.log({'Validation set prediction correlation': wandb.Image(tmp_file_path)})

    plot_correlation(test_predicted_values, test_true_values, file_path=tmp_file_path)
    wandb.log({'Test set prediction correlation': wandb.Image(tmp_file_path)})

    plot_target_histogram(train_true_values, val_true_values, test_true_values, file_path=tmp_file_path)
    wandb.log({'Target value distributions': wandb.Image(tmp_file_path)})

    # end run
    wandb.finish(exit_code=0)


    # Add the delay_seconds argument to run the jobs with a number
    # of seconds delay in between.

if __name__ == "__main__":

    general_path = '/home/gc_s40870_802050k_l3n1_ddcr_sum_mul_geom_depth2/'

    # run jobs sequentially
    schedule.every(20).minutes.do(run_job, wandbname='gc_s40870_802050k_l3n1_ddcr_sum_mul_geom_depth2', property='Gravimetric_Uptake_wt%_g_g', dnn=1)    

    schedule.run_all()
