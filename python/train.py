import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from python.data_loader import InclusiveDataset
import random 
import numpy as np
from models.EventClassifier import EventClassifier
import matplotlib.pyplot as plt
import mplcursors
from load_config import load_config
from datetime import datetime 
from comet_ml import Experiment
import argparse

# let' set comet 
def set_comet(config):
    base_name = config['project']['name']

    # get current time as string
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # final name
    run_name = f"{base_name}_{now}"

    experiment = Experiment()

    # set in comet
    experiment.set_name(run_name)

    return experiment, run_name

## a function to 
def parse_args():
    parser = argparse.ArgumentParser(description="Train Event Classifier")
    parser.add_argument('--config', type=str, default='configs/minimal_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    return args

def setrandom_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(config):
    ## let's load common things for config 
    config_dataset= config['dataset']
    event_input_features = config['model']['inputs']['event_features']
    object_input_features = config['model']['inputs']['object_features']
    class_label = config['model']['class_label']
    config_training_params = config['training']

    train_dataset = InclusiveDataset(config_dataset['train'], config_dataset['norm'], event_input_features, object_input_features, class_label)
    val_dataset = InclusiveDataset(config_dataset['val'], config_dataset['norm'], event_input_features, object_input_features, class_label)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config_training_params['batch_size'], shuffle=config_training_params['shuffle'], num_workers=config_training_params['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=config_training_params['batch_size'], shuffle=False, num_workers=config_training_params['num_workers'])
    
    return train_dataloader, val_dataloader

def get_model_paramaters(config):
    model_params = config['model']
    event_input_dim = len(model_params['inputs']['event_features'])
    object_input_dim = len(model_params['inputs']['object_features'])
    
    return {
        'event_input_dim': event_input_dim,
        'object_input_dim': object_input_dim,
        'hidden_dim': model_params['hidden_dim'],
        'output_dim': model_params['output_dim'],
        'num_heads': model_params['num_heads'],
        'num_encoder_layers': model_params['num_encoder_layers'],
        'ff_dim': model_params['ff_dim'],
        'batch_first': True
    }

## main function to run the training 

def main(config_path):
    config = load_config(config_path)

    # let's set comet here 
    experiment, model_name = set_comet(config)
    
    # set random seeds
    setrandom_seeds(config['training']['seed'])
    
    # get dataloaders
    train_dataloader, val_dataloader = get_dataloaders(config)
    
    # get model parameters
    model_params = get_model_paramaters(config)
    
    # initialize the model
    model = EventClassifier(**model_params)
    
    # define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=float(config['training']['weight_decay']))
    
    # do i plot locally?
    make_plots_locally = config['plots']['make_local_plots']
    path_to_save = config['plots']['plot_path']

    if make_plots_locally:
        plt.ion()  # enable interactive mode for live plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
        train_step_line, = ax1.plot([], [], label="Train loss", color="blue")
        val_step_line, = ax1.plot([], [], label="Val loss", color="orange")
        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax2.grid(True)

        train_epoch_line, = ax2.plot([], [], label="Train epoch loss", color="blue")
        val_epoch_line, = ax2.plot([], [], label="Val epoch loss", color="orange")

    # start the training loop now 
    num_epochs = config['training']['num_epochs']
    print("Starting Training Loop...")
    train_epoch_losses = []
    val_epoch_losses = []
    val_step_losses = []
    train_step_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (event_feats, object_feats, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(event_feats, object_feats)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_step_losses.append(loss.item())

            experiment.log_metric("train_batch_loss", loss.item(), step=epoch * len(train_dataloader) + batch_idx)

            #stop after first 3 batches for testing 
            if batch_idx >= 20:
                break
        
        avg_loss = train_loss / len(train_dataloader)
        train_epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
        experiment.log_metric("train_epoch_loss", avg_loss, step=epoch)

        if make_plots_locally:
            train_step_line.set_data(np.arange(len(train_step_losses)), train_step_losses)
        
        # validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (event_feats, object_feats, labels) in enumerate(val_dataloader):
                outputs = model(event_feats, object_feats)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()

                val_step_losses.append(loss.item())

                experiment.log_metric("val_batch_loss", loss.item(), step=epoch * len(val_dataloader) + batch_idx)
                #stop after first 3 batches for testing 
                if batch_idx >= 20:
                    break
        
            val_step_line.set_data(np.arange(len(val_step_losses)), val_step_losses)
       
        avg_val_loss = val_loss / len(val_dataloader)
        val_epoch_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
        experiment.log_metric("val_epoch_loss", avg_val_loss, step=epoch)

    if make_plots_locally:
        mplcursors.cursor([train_step_line,val_step_line], hover=True)
        ax1.set_title("Training & Validation Loss Monitoring")
        ax1.set_xlabel("Batch step")
        ax1.set_ylabel("Loss")
        ax1.relim()
        ax1.autoscale_view()
        plt.pause(0.01)

        # update epoch plot
        train_epoch_line.set_data(np.arange(len(train_epoch_losses)), train_epoch_losses)
        val_epoch_line.set_data(np.arange(len(val_epoch_losses)), val_epoch_losses)
        mplcursors.cursor([train_epoch_line,val_epoch_line], hover=True)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.relim()
        ax2.autoscale_view()
        plt.pause(0.1)

        plt.ioff()  # disable interactive mode
        plt.savefig(path_to_save+"/Train_Validation_Loss_"+model_name+".png")
        plt.show()  # show the final plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    main(args.config)