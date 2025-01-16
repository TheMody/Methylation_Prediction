import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

#tunable hyperparameters (now they are getting set with optuna)
# lr = 1e-4
# lr_cls = 1e-4
# epochs_cls = 100
# dim_hidden = 256
# num_blocks = 4
# compression = 32
# model_type  = "transformer"
# mask_ratio = 0.15

#non tunable hyperparamters
batch_size = 32
batch_size_cls = 8
gradient_accumulation_steps = 1
num_classes = 3
pad_size = 27584
num_inputs = pad_size
num_inputs_original = 27578
seed = 42
folder = "experiment_1"

config = {
    'epochs': epochs,
    "pad_size": pad_size,
    'seed': seed,
    'device': device,
    'num_classes': num_classes,
    'num_inputs': num_inputs,
    'batch_size': batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
}