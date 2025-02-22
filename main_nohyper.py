from train import train
from pre_train import pre_train
import optuna
import pickle
from config import folder
import os
if __name__ == "__main__":
        
    os.makedirs(folder, exist_ok=True)
    hyperparameters = {
        "lr_cls": 2e-5,
        "lr": 1e-4,
        "epochs_cls": 100,
        "dim_hidden": 256,
        "num_blocks": 4,
        "compression": 32,
        "model_type": "transformer",
        "mask_ratio": 0.15,
        "epochs":  10,
        }
    model = pre_train(hyperparameters)
    scores = train(hyperparameters,model )
    print(scores)
    

