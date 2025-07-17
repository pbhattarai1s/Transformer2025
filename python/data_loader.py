import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class InclusiveDataset(Dataset):
    #class initualization
    def __init__(self, h5_file, norm_file, event_input_features=None, object_input_features=None, class_label=''):
        self.h5_file = h5py.File(h5_file, 'r')
        all_events = self.h5_file['events'][:]
        all_objects = self.h5_file['objects'][:]

        self.valid_mask = all_objects['valid']
        self.mbb = all_events['mBB']

        # Select only specified columns
        self.events = all_events[event_input_features] if event_input_features else all_events
        self.objects = all_objects[object_input_features] if object_input_features else all_objects
        self.labels = all_events[class_label]

        if norm_file is None:
            raise ValueError("Normalization file path must be provided.")
        
        with open(norm_file, 'r') as f:
            norm_params = yaml.safe_load(f)

        self.event_means = np.array([norm_params['events'][name]['mean'] for name in self.events.dtype.names ])
        self.event_stds = np.array([norm_params['events'][name]['std'] for name in self.events.dtype.names ])
        self.object_means = np.array([norm_params['objects'][col]['mean'] for col in self.objects.dtype.names])
        self.object_stds = np.array([norm_params['objects'][col]['std'] for col in self.objects.dtype.names])
        self.safe_object_stds = np.where(self.object_stds > 0, self.object_stds, 1.0) # to avoid division by zero we replace 0 with 1.0

   
    #length of events
    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        #object_features = self.objects[idx]

        # Normalize event features
        event_features = np.array([event[name] for name in self.events.dtype.names ])
        event_features = (event_features - self.event_means) / self.event_stds

        # extract object data
        object_data = np.array([[self.objects[idx, obj_idx][name] for name in self.objects.dtype.names] for obj_idx in range(self.objects.shape[1])])
        object_data = (object_data - self.object_means) / self.safe_object_stds 

        event_tensor = torch.tensor(event_features, dtype=torch.float32)
        object_tensor = torch.tensor(object_data, dtype=torch.float32) 
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.int64)
        #valid_mask = np.array([self.valid[idx, obj_idx] for obj_idx in range(self.objects.shape[1])], dtype=bool)
        valid_mask_tensor = torch.tensor(self.valid_mask[idx], dtype=torch.bool)
        mbb_tensor = torch.tensor(self.mbb[idx], dtype=torch.float32)

        return event_tensor, object_tensor, label_tensor, valid_mask_tensor, mbb_tensor
    
