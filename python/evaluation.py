import yaml 
import torch 
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.EventClassifier import EventClassifier
from python.data_loader import InclusiveDataset
from load_config import load_config
import random
from torch.utils.data import DataLoader
import numpy as np

from python.train import get_model_paramaters, setrandom_seeds
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import h5py


"""
This takes a model, a test file and optionally a checkpoint for evaluation
"""

def parse_args():
    parser = argparse.ArgumentParser(description=" evaluate model")
    parser.add_argument('--model_name', type=str, required=True, help = "Model name ")
    parser.add_argument('--test_file',type=str, required=True, help = "test file ")
    parser.add_argument('--ckpt', type=str, required=False, help = "specific model ckpt for evaluation" )
    args = parser.parse_args()

    return args

def getckpt(path,ckpt=None):
    full_path = os.path.join(path, "ckpt") 
    if ckpt:
        return os.path.join(full_path,ckpt)
    all_ckpts = os.listdir(full_path)
    best_ckpt = None
    best_loss =  float('inf')
    for ckpt in all_ckpts:
        if not ".pth" in ckpt:
            continue
        loss =  float(ckpt.split("val_loss_")[-1].strip(".pth"))
        if loss < best_loss:
            best_loss = loss
            best_ckpt = ckpt
    return os.path.join(full_path,best_ckpt)

def get_test_dataloaders(config, test_file):
    config_dataset= config['dataset']
    event_input_features = config['model']['inputs']['event_features']
    object_input_features = config['model']['inputs']['object_features']
    class_label = config['model']['class_label']
    config_training_params = config['training']

    # per event train dataset
    test_dataset = InclusiveDataset(test_file, config_dataset['norm'], event_input_features, object_input_features, class_label)

    #data loader for test disabled shuffle for reproducibility
    test_dataloader = DataLoader(test_dataset, batch_size=config_training_params['batch_size'], shuffle=False, num_workers=config_training_params['num_workers'])

    return test_dataloader

def save_scores_to_h5(input_h5_file, output_h5_file, output_scores, pred_adv_scores):
    # Open the input file
    with h5py.File(input_h5_file, "r") as fin:
        # Copy the objects dataset directly
        objects_data = fin["objects"][:]
        events_data = fin["events"][:]
        
    # Make sure the length matches
    assert len(events_data) == len(output_scores), "Length mismatch!"

    # Build new dtype with extra field
    old_dtype = events_data.dtype
    new_dtype = old_dtype.descr + [("classifier_scores", np.float32)]
    new_dtype = new_dtype + [("adverserial_scores", np.int32)]
    
    # Create new structured array
    new_events_data = np.empty(events_data.shape, dtype=new_dtype)
    for name in old_dtype.names:
        new_events_data[name] = events_data[name]
    new_events_data["classifier_scores"] = output_scores
    new_events_data["adverserial_scores"] = pred_adv_scores

    # Now write to output file
    with h5py.File(output_h5_file, "w") as fout:
        fout.create_dataset("objects", data=objects_data)
        fout.create_dataset("events", data=new_events_data)

    print(f"Saved predictions to {output_h5_file}")


def evaluate(model_param, ckpt_path, test_loader, label_weight=0):
    # let's get the model
    model = EventClassifier(**model_param)
    # let's load the state (weights, paramaters) of the model from a ckpt
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #set model in evaluation mode
    model.eval()
    # some storage
    total_class_loss, total_adv_loss, total_output_loss = 0.0, 0.0, 0.0
    all_labels, all_class_probs, all_adv_preds = [],[], []

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([label_weight]))
    criterion_adv = torch.nn.CrossEntropyLoss()

    #let's switch torch to evaluation mode dropping batchnorm, dropout..
    with torch.no_grad():
        for batch_idx, (event_feats, object_feats, labels, valid_mask, mbb) in enumerate(test_loader):
            mbb_bins = torch.bucketize(mbb / 1000.0, torch.tensor(model_param['mbb_bins'], device=mbb.device)) - 1
            class_logits, adv_logits = model(event_feats, object_feats, valid_mask)
        
            class_probs = torch.sigmoid(class_logits).squeeze()
            adv_probs = torch.softmax(adv_logits, dim=1) #this changes 
            adv_preds = torch.argmax(adv_probs, dim=1)
    
            all_class_probs.append(class_probs)
            all_adv_preds.append(adv_preds)
            all_labels.append(labels)
            
            # i don't think we need loss
            #loss = criterion(class_outputs.squeeze(), labels.float())
            #total_class_loss += loss.item()

            # loss_adv = criterion_adv(adv_output, mbb_bins)
            # total_adv_loss += loss_adv.item()
            # loss += model_param['adv_lambda'] * loss_adv
            # total_output_loss += loss.item()
            
            # stop after first 3 batches for testing 
            # if batch_idx >= 500:
            #     break
        
    all_class_probs = torch.cat(all_class_probs)
    all_adv_preds = torch.cat(all_adv_preds)
    all_labels = torch.cat(all_labels)

    return all_labels, all_class_probs, all_adv_preds


def plot_roc_curve( labels, preds, save_path):
    
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.savefig(save_path+"/roc_curve.png")
    plt.close() 

def main(model, test_file, ckpt=None):
    model_path = os.path.join("logs",model)
    
    # let's load config
    config = load_config(model_path + "/config.yaml")
    #set random seeds for reproducibility
    setrandom_seeds(config['training']['seed'])
    # get ckpt to evaluate
    eval_ckpt = getckpt(model_path, ckpt)
    # get train dataloaders 
    test_dataloader = get_test_dataloaders(config,test_file)
    # get model param from train.py 
    model_params = get_model_paramaters(config)
    # evaluate model 
    true_label, pred_class_scores, pred_adv_scores  = evaluate(model_params,eval_ckpt,test_dataloader)

    file_output_path = os.path.join(eval_ckpt.split(".pth")[0]+"___"+str(test_file).split("/")[-1])

    print(file_output_path)

    save_scores_to_h5(test_file, file_output_path, pred_class_scores,pred_adv_scores )
    
    #path for saving roc curve
    roc_path = os.path.join(config['plots']['plot_path'], model)

    plot_roc_curve(true_label,pred_class_scores,roc_path)

if __name__ == "__main__":
    args = parse_args()
    
    main(args.model_name, args.test_file, args.ckpt)

