import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
lr = 1e-4
epochs = 10
batch_size = 32
dim_hidden = 256
num_classes = [3]
num_inputs = 27578
num_blocks = 4
compression = 32
model_type  = "mlp"
config = {
    'device': device,
    'lr': lr,
    'epochs': epochs,
    'batch_size': batch_size,
    'dim_hidden': dim_hidden,
    'num_classes': num_classes,
    'num_inputs': num_inputs,
    'num_blocks': num_blocks,
    'compression': compression,
    'model_type': model_type
}