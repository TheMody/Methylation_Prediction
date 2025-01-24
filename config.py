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
#running mean tensor([1495.5520, 1878.5018, 1144.4940,  ...,    0.0000,    0.0000,0.0000])
#running var tensor([1.0979e+03, 1.4760e+03, 8.9545e+02,  ..., 1.0000e+00, 1.0000e+00, 1.0000e+00]) for GSE13204
#running mean tensor([444.7763, 444.3635, 282.9825,  ...,   0.0000,   0.0000,   0.0000])
#running var tensor([1.9517e+03, 2.5516e+03, 2.1031e+03,  ..., 1.0000e+00, 1.0000e+00,1.0000e+00]) for whole gpl570
#non tunable hyperparamters
batch_size = 16
batch_size_cls = 8
gradient_accumulation_steps = 2
num_classes = 18 #165
pad_size = 54784#482304 #27584 #
num_inputs = pad_size
num_inputs_original = 54675#for gpl570 481868 for 470k illumina#27578  for 27k illumino#
seed = 42
folder = "experiment_1"
path_to_data = "/media/philipkenneweg/Data/datasets/GPL570/" #"methylation_data/"#

config = {
    "pad_size": pad_size,
    'seed': seed,
    'device': device,
    'num_classes': num_classes,
    'num_inputs': num_inputs,
    'batch_size': batch_size,
    "batch_size_cls": batch_size_cls,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "folder": folder,
}