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

from comet_ml import Experiment
experiment = Experiment()

## set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

#create dataset first 
train_dataset = InclusiveDataset('outputh5/train.h5', 'outputh5/norm.yaml')
val_dataset = InclusiveDataset('outputh5/val.h5', 'outputh5/norm.yaml')

# wrap in dataloader 
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# let's initialize the model
model = EventClassifier(
    event_input_dim=len(train_dataset.events.dtype.names) - 1,  # excluding label
    object_input_dim=len(train_dataset.objects.dtype.names),
    hidden_dim=64,
    output_dim=64,  # assuming binary classification
    num_heads=4,
    num_encoder_layers=6,
    ff_dim=128,
    batch_first=True
)

#define the loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
# --------- ACTUAL TRAINING LOOP ---------
plt.ion()  # enable interactive mode for live plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
train_step_line, = ax1.plot([], [], label="Train loss", color="blue")
val_step_line, = ax1.plot([], [], label="Val loss", color="orange")
ax1.legend(loc='upper right')
ax1.grid(True)
ax2.grid(True)

train_epoch_line, = ax2.plot([], [], label="Train epoch loss", color="blue")
val_epoch_line, = ax2.plot([], [], label="Val epoch loss", color="orange")

print("Starting Training Loop...")
train_epoch_losses = []
val_epoch_losses = []
val_step_losses = []
train_step_losses = []
# loop over epochs 
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    # Training loop
    for batch_idx, (event_feats, object_feats, labels) in enumerate (train_dataloader):
        optimizer.zero_grad()
        outputs = model(event_feats, object_feats)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_step_losses.append(loss.item())

        # comet logging
        experiment.log_metric("train_batch_loss", loss.item(), step=epoch * len(train_dataloader) + batch_idx)

        
        #stop after first 3 batches for testing 
        # if batch_idx >= 20:
        #     break
    
    # update step plot every few batches for speed
    #ax1.plot(train_step_losses, color='blue')
    train_step_line.set_data(np.arange(len(train_step_losses)), train_step_losses)
    
    avg_loss = train_loss / len(train_dataloader)
    train_epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
    experiment.log_metric("train_epoch_loss", avg_loss, step=epoch)

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
            # if batch_idx >= 20:
            #     break
        
        val_step_line.set_data(np.arange(len(val_step_losses)), val_step_losses)
    mplcursors.cursor([train_step_line,val_step_line], hover=True)
    ax1.set_title("Training & Validation Loss Monitoring")
    ax1.set_xlabel("Batch step")
    ax1.set_ylabel("Loss")
    ax1.relim()
    ax1.autoscale_view()
    plt.pause(0.01)

    avg_val_loss = val_loss / len(val_dataloader)
    val_epoch_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
    experiment.log_metric("val_epoch_loss", avg_val_loss, step=epoch)

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
plt.show()  # show the final plot

