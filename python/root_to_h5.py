import uproot
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

# Define the features to extract from the event and objects
# might makes sense to add this later to a config file
# or a json file, but for now we will keep it simple
event_features_to_extract = [
    "mBB",
    "pTBB",
    "eventWeight",
    "nJets20pt_central",
    "nJets20pt_rap",
    "dPhiBB",
    "mJJ",
    "dPhiJJ",
    "dPhiBBJJ",
    "dEtaJJ",
    "dEtaBBJJ",
    "pT_balance",
    "centralMaxPt",
    "mBB_angle",
    "asymJJ",
]
object_features_to_extract = [
    ["pTJ1", "pTJ2"],
    ["etaJ1", "etaJ2"],
    ["GN2v01bBinB1", "GN2v01bBinB2"],
    ["GN2v01bBinJ1", "GN2v01bBinJ2"],
    ["GN2v01cBinB1", "GN2v01cBinB2"],
    ["GN2v01cBinJ1", "GN2v01cBinJ2"],
    ["NTrk500PVJ1", "NTrk500PVJ2"],
    ["QGTransformer_ConstScoreJ1", "QGTransformer_ConstScoreJ2"]
]
# define a function to load the root file and extract the features 
def load_root_file(file_path, filename, label ):
    with uproot.open(file_path+filename) as file:
        tree = file["Nominal"]
        if "mcChannelNumber" in tree.keys(): #this is where we can introduce more background later , by adding mc channel number as function input
            print("mcChannelNumber found in tree keys.")
    event_feature_array = np.array([tree[feature].array(library="np") for feature in event_features_to_extract]).T
    print("event_feature_array shape:", event_feature_array.shape)
    object_feature_array = np.array([[tree[feature].array(library="np") for feature in features] for features in object_features_to_extract]).T
    # create label for the event 
    labels = np.full((event_feature_array.shape[0],), label, dtype=np.int8)
    return event_feature_array, object_feature_array, labels
# should also add file_path and filename in the config file later
# for now we will keep it simple
# call the function to see if it works 
filepath = "~/Documents/VBFHbbHcc/fitinputs20250502/BB/"
files = [
    {"filename": "tree_vbfhbb18.root", "label": 1},
    {"filename": "tree_ggfhbb18.root", "label": 1},
    {"filename": "tree_data18.root", "label": 0}
]
all_event_features = np.empty((0, len(event_features_to_extract)))
all_object_features = np.empty((0, 2, len(object_features_to_extract)))  # 2 for each object feature pair
all_labels = np.empty((0,), dtype=np.int8)
for file in files:
    event_feature_array, object_feature_array, labels =load_root_file(filepath, file["filename"], file["label"])
    # get all features 
    all_event_features = np.vstack((all_event_features, event_feature_array))
    all_object_features = np.vstack((all_object_features, object_feature_array))
    all_labels = np.hstack((all_labels, labels))

# let's shuffle the arrays befiore saving them 
# get total number of events
N_total = all_labels.shape[0]

# make indices
indices = np.arange(N_total)
np.random.shuffle(indices)

# apply shuffle to all arrays, this is for sanity check, can also skip and just use the shuggle from the train_test_split
all_event_features = all_event_features[indices]
all_object_features = all_object_features[indices]
all_labels = all_labels[indices]

## to be stores in config file later
train_val_test_split = [0.8,0.1,0.1]  # 80% train, 10% validation, 10% test
random_state = 42  # for reproducibility , we will also keep this in config file later

# split the data into train, validation and test sets
X_train_event_feature, X_temp_event_feature, X_train_object_feature, X_temp_object_feature, y_train, y_temp = train_test_split(all_event_features, all_object_features, all_labels, random_state=random_state, test_size=1-train_val_test_split[0], stratify=all_labels, shuffle=True)
X_val_event_feature, X_test_event_feature, X_val_object_feature, X_test_object_feature, y_val, y_test = train_test_split(X_temp_event_feature, X_temp_object_feature, y_temp, random_state=random_state, test_size=train_val_test_split[2]/(train_val_test_split[1]+train_val_test_split[2]), stratify=y_temp, shuffle=True)

## -- now let's save the data to h5 files -- ##
def save_to_h5( X_event, X_object, y, filename):
    # let me first create a data type for event features 
    event_dtype = [(name, np.float32) for name in event_features_to_extract] + [('label', np.int8)]
    # create a structured array for event features
    structured_events = np.zeros(X_event.shape[0], dtype=event_dtype)
    for idx, name in enumerate(event_features_to_extract):
        structured_events[name] = X_event[:, idx]
    structured_events['label'] = y
    
    # now let's do the same for object features
    object_columns = ["VBFjetPt", "VBFjetEta", "GN2v01BinBJet", "GN2v01BinVBfJet", "GN2v01cBinBJet", "GN2v01cBinVBfJet", "NTrk500PVVBFJet", "QGTransformer_ConstScoreVBFJet" ]
    # let's add some additional placeholders for additional objects in an event 
    N_events, N_objects, N_features = X_object.shape
    N_max_objects = 5  # assuming we have at most 2 objects per event
    X_object_padded = np.full((N_events, N_max_objects, N_features), 0.0, dtype=np.float32)  # fill with NaN for missing objects
    X_object_padded[:, :N_objects, :] = X_object
    # let me create a structured array for object features
    object_dtype = [(col, np.float32) for col in object_columns]
    structured_objects = np.zeros((N_events,N_max_objects), dtype=object_dtype)
    print("structured_objects shape:", structured_objects.shape)
    for idx, col in enumerate(object_columns):
        structured_objects[col] = X_object_padded[:,:, idx] 

    # ****** ------ only save normalization parameters for training set ------ ****** #
    if ("train" in filename):
        ## before saving the h5 file, let's also save normalization parameters for the features for training set
        event_norm_params = {}
    
        for idx, name in enumerate(event_features_to_extract):
            mean = np.mean(X_event[:, idx])
            std = np.std(X_event[:, idx])
            event_norm_params[name] = {'mean': float(mean), 'std': float(std)}
    
        object_norm_params = {}
        for idx, col in enumerate(object_columns):
            values = X_object[:, :, idx].flatten()
            object_norm_params[col] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
        
        class_counts = {int(cls): int(np.sum(y_train == cls)) for cls in np.unique(y_train)}

        yaml_data = {
            'events': event_norm_params,
            'objects': object_norm_params,
            "labels": {"counts": class_counts}
        }
        with open("norm.yaml", "w") as f:
            yaml.dump(yaml_data, f)
        
    # ****** ------ save h5s ------ ****** #
    with h5py.File(filename, 'w') as f:
        f.create_dataset('events', data=structured_events)
        f.create_dataset('objects', data=structured_objects)
    print(f"Data saved to {filename}")

## save all the data to h5 files
save_to_h5(X_train_event_feature, X_train_object_feature, y_train, 'train.h5')
save_to_h5(X_val_event_feature, X_val_object_feature, y_val, 'val.h5')
save_to_h5(X_test_event_feature, X_test_object_feature, y_test, 'test.h5')
