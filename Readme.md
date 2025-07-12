# VBF Higgs ML Classification Framework

This is a lightweight ML pipeline designed to explore VBF Higgs → b b̄ / c c̄ signals vs background at the LHC.  
It is built entirely from scratch in PyTorch with a YAML-based configuration system and works on structured event & object data from h5 files.

This is basically my obsession with trying to fully understand what goes in building a ML pipeline. 

This is outcome of 5 days staycation slow/self-learning pace. 

---

## Environment 
There are currently two environment used. <span style="color:red"> To do: add env files to autogenerate env</span>

- ml_env : to run the training, evaluation and plotting from h5s
```bash
    conda create -n ml_env python=3.10
    conda activate ml_env
    pip install torch numpy matplotlib
    conda install h5py
    pip install pyyaml
    Pip install mplcursors
    pip install comet_ml
    conda install scikit-learn
```
- root_to_h5_env : to run the training, evaluation and plotting from h5s
```bash
        conda create -n root_to_h5_env python=3.10 numpy h5py
        conda activate root_to_h5_env
        pip install uproot awkward
        Pip install pyyaml
    
```

## How to Run

- Training 
```bash
python train.py --config configs/minimal_config.yaml
```
- Evaluation 
```bash
ython evaluate.py --model_name <log_dir> --test_file test.h5
```
Can also pass an optional ckpt incase to test a particular ckpt.


## Features
- Processes **event-level features** through an MLP.
- Processes **object-level features** (events, objects, etc) through a Transformer Encoder.
- Supports **masked attention** to ignore padded objects.
- Combines event & object representations for binary classification.
- Fully configurable via YAML.
- Trains & evaluates on H5 datasets, automatically normalizes inputs.
- Writes output scores back to a copy of `test.h5` for downstream studies.

---

## Main Structure
- configs 
    stored config files in `*.yaml` to steer the config
- models 
    where all the models will be stored. currently has event, object and overall classifier 
- python 
    - data_loader.py # InclusiveDataset handles reading & normalization
    - train.py # trains model, logs to comet, saves ckpts & plots,
    - evaluate.py # loads model, predicts on test set, updates H5
    - root_to_h5.py # convert root to h5 files with "events", "objects" structure; event has a scalar dattype and object is a vector 
    - load_config.py # parses yaml

## To DO and Planned Extensions 

## Physics / ML improvements
- [ ] Add decorrelation loss (CLR / uniformization or adversarial) to reduce correlation with mBB
- [ ] Implement multi-class / multi-task mode from config (e.g. Hbb vs Hcc vs Bkg)
- [ ] Add multi-head outputs for nuisance or auxiliary tasks
- [ ] Improve object masking (try learnable mask embeddings inside attention)

## Postprocessing / analysis
- [ ] Plot confusion matrix, ROC by class, AUC vs epoch
- [ ] Plot histograms of output scores split by true label
- [ ] Scan for optimal signal significance
- [ ] Dump kinematic variables of highest scoring events for sanity checks

##  Code infrastructure
- [ ] Add GPU support (`.to(device)` for all tensors, model, loss)
- [ ] General code cleanup & consistent naming (objects vs events etc)
- [ ] Move root→h5 to config-driven pipeline (features, input files, splits)
- [ ] Log comet experiment URLs for traceability
- [ ] Make tasks, loss functions and various other things fully configurable from .yaml

##  Optional next steps
- [ ] Run hyperparameter optimization (Optuna or comet sweeps)
- [ ] Try advanced architectures (SetTransformer, Graph Networks)

