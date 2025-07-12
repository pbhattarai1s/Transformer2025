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
import shutil

"""
This is where all the magic happens!

We get a config file, parse it, get dataloaders (aka tensors for training)

get the event classifier model, and train 

save model after every epoch, plot losses locally and also log in comet!

"""

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
    val_dataloader = DataLoader(val_dataset, batch_size=config_training_params['batch_size'], shuffle=False, num_workers=config_training_params['num_workers']) #disable shuffle for reproducibility
    
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

def save_checkpoint( model, optimizer, epoch, loss, path):
    state = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state,path)

## main function to run the training 
def main(config_path):
    config = load_config(config_path)

    # let's set comet here 
    experiment, model_name = set_comet(config)

    # let's create some output paths and folders to save stuff 
    os.makedirs(os.path.join("logs",model_name, "ckpt"))
    shutil.copy(config_path, os.path.join("logs",model_name, "config.yaml"))
    # set random seeds
    setrandom_seeds(config['training']['seed'])
    
    # get dataloaders
    train_dataloader, val_dataloader = get_dataloaders(config)
    
    # get model parameters
    model_params = get_model_paramaters(config)
    
    # initialize the model
    model = EventClassifier(**model_params)
    
    # define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['model']['pos_label_weight']]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=float(config['training']['weight_decay']))
    
    # do i plot locally?
    make_plots_locally = config['plots']['make_local_plots']
    path_to_save = config['plots']['plot_path']
    os.makedirs(os.path.join(path_to_save, model_name))

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
        
        for batch_idx, (event_feats, object_feats, labels,valid_mask) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # event_feats = torch.randn_like(event_feats)
            #object_feats = torch.randn_like(object_feats)
            #labels = torch.randint(0, 2, labels.shape, dtype=labels.dtype)
            outputs = model(event_feats, object_feats, valid_mask)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_step_losses.append(loss.item())

            experiment.log_metric("train_batch_loss", loss.item(), step=epoch * len(train_dataloader) + batch_idx)

            #stop after first 3 batches for testing 
            # if batch_idx >= 20:
            #     break

            #if epoch == 0 and batch_idx == 0:
            #print("epoch: ", epoch, "batch indx: ",batch_idx , f"Batch event_feats mean/std: {event_feats.mean():.3f}/{event_feats.std():.3f}")
            #print(f"Batch object_feats mean/std: {object_feats.mean():.3f}/{object_feats.std():.3f}")
            # print(f"Outputs logits mean/std: {outputs.mean():.3f}/{outputs.std():.3f}")
            # print(f"Outputs logits mean/std labels == 1: {outputs[labels==1].mean():.3f}/{outputs[labels==1].std():.3f}")
            # print(f"Outputs logits mean/std labels == 0: {outputs[labels==0].mean():.3f}/{outputs[labels==0].std():.3f}")
        
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
            for batch_idx, (event_feats, object_feats, labels, valid_mask) in enumerate(val_dataloader):
                outputs = model(event_feats, object_feats, valid_mask)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()

                val_step_losses.append(loss.item())

                experiment.log_metric("val_batch_loss", loss.item(), step=epoch * len(val_dataloader) + batch_idx)
                #stop after first 3 batches for testing 
                # if batch_idx >= 20:
                #     break
        
            val_step_line.set_data(np.arange(len(val_step_losses)), val_step_losses)
       
        avg_val_loss = val_loss / len(val_dataloader)
        val_epoch_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
        experiment.log_metric("val_epoch_loss", avg_val_loss, step=epoch)

         # save after each epoch
        checkpoint_path = os.path.join("logs", model_name, "ckpt", f"epoch_{epoch+1:02d}_val_loss_{avg_val_loss:.5f}.pth")
        print (checkpoint_path)
        save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_path)
        

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
        plt.savefig(path_to_save+model_name+"/Train_Validation_Loss.png")
        plt.show()  # show the final plot

if __name__ == "__main__":
    args = parse_args()
    main(args.config)