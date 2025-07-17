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
from matplotlib.lines import Line2D
import mplcursors
from load_config import load_config
from datetime import datetime 
from comet_ml import Experiment
import argparse
import shutil
from train_single_epoch import run_epoch

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
        'batch_first': True,
        'adv_lambda':  model_params['adv_lambda'],
        'adv_method': model_params['decorrelation_strategy'], 
        'mbb_bins': model_params['mbb_bins']
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
    criterion_adv = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=float(config['training']['weight_decay']))
    
    # do i plot locally?
    make_plots_locally = config['plots']['make_local_plots']
    path_to_save = config['plots']['plot_path']
    os.makedirs(os.path.join(path_to_save, model_name))

    if make_plots_locally:
        plt.ion()  # enable interactive mode for live plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

         #Create custom legend handles
        legend_lines = [
            Line2D([0], [0], color="black", linestyle='--', label="classifier loss"),
            Line2D([0], [0], color="black", linestyle='-.', label="adversary loss"),
            Line2D([0], [0], color="black", linestyle='-', label="total loss")
        ]

        # Add a global legend to the whole figure
        fig.legend(handles=legend_lines, loc='upper center', ncol=3, frameon=False)
    
        train_step_line_class, = ax1.plot([], [], label="Train loss", color="blue",linestyle='--' )
        val_step_line_class, = ax1.plot([], [], label="Val loss", color="orange", linestyle='--')
        train_step_line_advers, = ax1.plot([], [], color="blue", linestyle='-.')
        val_step_line_advers, = ax1.plot([], [], color="orange", linestyle='-.')
        train_step_line_tot, = ax1.plot([], [], color="blue")
        val_step_line_tot, = ax1.plot([], [], color="orange")
        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax2.grid(True)

        train_epoch_line_class, = ax2.plot([], [], label="Train loss", color="blue",linestyle='--' )
        val_epoch_line_class, = ax2.plot([], [], label="Val loss", color="orange", linestyle='--')
        train_epoch_line_advers, = ax2.plot([], [], color="blue", linestyle='-.')
        val_epoch_line_advers, = ax2.plot([], [], color="orange", linestyle='-.')
        train_epoch_line_tot, = ax2.plot([], [], color="blue")
        val_epoch_line_tot, = ax2.plot([], [], color="orange")

    # start the training loop now 
    num_epochs = config['training']['num_epochs']
    print("Starting Training Loop...")
    # save losses for plotting
    train_epoch_losses_classifier, val_epoch_losses_classifier,val_step_losses_classifier, train_step_losses_classifier = [], [] ,[] ,[]

    train_epoch_losses_advers, val_epoch_losses_advers, val_step_losses_advers, train_step_losses_advers = [], [] ,[] ,[]

    train_epoch_losses_tot, val_epoch_losses_tot,val_step_losses_tot, train_step_losses_tot = [], [] ,[] ,[]


    for epoch in range(num_epochs):
        model.train()
        torch.set_grad_enabled(True)

        # === TRAINING ===
        avg_train_clf, avg_train_adv, avg_train_tot, clf_steps, adv_steps, tot_steps = run_epoch(
            model=model,
            dataloader=train_dataloader,
            model_params=model_params,
            criterion=criterion,
            criterion_adv=criterion_adv,
            experiment=experiment,
            epoch=epoch,
            is_train=True,
            optimizer=optimizer,
            #max_batches=3  # optional: remove for full training
        )

        train_epoch_losses_classifier.append(avg_train_clf)
        train_epoch_losses_advers.append(avg_train_adv)
        train_epoch_losses_tot.append(avg_train_tot)

        print(f"[Epoch {epoch+1}] Train Losses — Classifier: {avg_train_clf:.4f}, Adversary: {avg_train_adv:.4f}, Total: {avg_train_tot:.4f}")
        experiment.log_metric("train_epoch_classifier_loss", avg_train_clf, step=epoch)
        experiment.log_metric("train_epoch_adverserial_loss", avg_train_adv, step=epoch)
        experiment.log_metric("train_epoch_total_loss", avg_train_tot, step=epoch)

        train_step_losses_classifier.extend(clf_steps)
        train_step_losses_advers.extend(adv_steps)
        train_step_losses_tot.extend(tot_steps)

        # Optional: update plot lines if you're plotting interactively
        if make_plots_locally:
            train_step_line_class.set_data(np.arange(len(train_step_losses_classifier)), train_step_losses_classifier)
            train_step_line_advers.set_data(np.arange(len(train_step_losses_advers)), train_step_losses_advers)
            train_step_line_tot.set_data(np.arange(len(train_step_losses_tot)), train_step_losses_tot)

        # === VALIDATION ===
        model.eval()
        torch.set_grad_enabled(False)

        avg_val_clf, avg_val_adv, avg_val_tot, val_clf_steps, val_adv_steps, val_tot_steps = run_epoch(
            model=model,
            dataloader=val_dataloader,
            model_params=model_params,
            criterion=criterion,
            criterion_adv=criterion_adv,
            experiment=experiment,
            epoch=epoch,
            is_train=False,
            #max_batches=3
        )

        val_epoch_losses_classifier.append(avg_val_clf)
        val_epoch_losses_advers.append(avg_val_adv)
        val_epoch_losses_tot.append(avg_val_tot)

        print(f"[Epoch {epoch+1}] val Losses — Classifier: {avg_val_clf:.4f}, Adversary: {avg_val_adv:.4f}, Total: {avg_val_tot:.4f}")
        experiment.log_metric("val_epoch_classifier_loss", avg_val_clf, step=epoch)
        experiment.log_metric("val_epoch_adverserial_loss", avg_val_adv, step=epoch)
        experiment.log_metric("val_epoch_total_loss", avg_val_tot, step=epoch)

        val_step_losses_classifier.extend(val_clf_steps)
        val_step_losses_advers.extend(val_adv_steps)
        val_step_losses_tot.extend(val_tot_steps)

        # Optional: update plot lines if you're plotting interactively
        if make_plots_locally:
            val_step_line_class.set_data(np.arange(len(val_step_losses_classifier)), val_step_losses_classifier)
            val_step_line_advers.set_data(np.arange(len(val_step_losses_advers)), val_step_losses_advers)
            val_step_line_tot.set_data(np.arange(len(val_step_losses_tot)), val_step_losses_tot)


        # Save checkpoint
        checkpoint_path = os.path.join("logs", model_name, "ckpt", f"epoch_{epoch+1:02d}_val_loss_{avg_val_tot:.6f}.pth")
        save_checkpoint(model, optimizer, epoch, avg_val_tot, checkpoint_path)

    if make_plots_locally:
        mplcursors.cursor([train_step_line_class,val_step_line_class], hover=True)
        mplcursors.cursor([train_step_line_advers,val_step_line_advers], hover=True)
        mplcursors.cursor([train_step_line_tot,val_step_line_tot], hover=True)
        ax1.set_title("Training & Validation Loss Monitoring")
        ax1.set_xlabel("Batch step")
        ax1.set_ylabel("Loss")
        ax1.relim()
        ax1.autoscale_view()
        ax1.set_yscale('log')
        plt.pause(0.01)

        # update epoch plot
        train_epoch_line_class.set_data(np.arange(len(train_epoch_losses_classifier)),train_epoch_losses_classifier)
        val_epoch_line_class.set_data(np.arange(len(val_epoch_losses_classifier)),val_epoch_losses_classifier)
        train_epoch_line_advers.set_data(np.arange(len(train_epoch_losses_advers)),train_epoch_losses_advers)
        val_epoch_line_advers.set_data(np.arange(len(val_epoch_losses_advers)),val_epoch_losses_advers)
        train_epoch_line_tot.set_data(np.arange(len(train_epoch_losses_tot)),train_epoch_losses_tot)
        val_epoch_line_tot.set_data(np.arange(len(val_epoch_losses_tot)),val_epoch_losses_tot)

       
        mplcursors.cursor([train_epoch_line_class,val_epoch_line_class], hover=True)
        mplcursors.cursor([train_epoch_line_advers,val_epoch_line_advers], hover=True)
        mplcursors.cursor([train_epoch_line_tot,val_epoch_line_tot], hover=True)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.relim()
        ax2.autoscale_view()
        ax2.set_yscale('log')
        plt.pause(0.1)

        plt.ioff()  # disable interactive mode
        plt.savefig(path_to_save+model_name+"/Train_Validation_Loss.png")
        plt.show()  # show the final plot

if __name__ == "__main__":
    args = parse_args()
    main(args.config)