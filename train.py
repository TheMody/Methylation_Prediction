import torch
import numpy as np
from model import MethylMLP, EncoderModelPreTrain

from config import *
from dataset import Methylation_ds
import wandb
from cosine_scheduler import CosineWarmupScheduler
import time
from tqdm import tqdm
import math

def train(hyperparameters,model = None):
    if model is None:
        if hyperparameters["model_type"] == "transformer":
            model = EncoderModelPreTrain(num_classes=num_classes, num_tokens=num_inputs, hidden_dim=hyperparameters["dim_hidden"], n_layers=hyperparameters["num_blocks"], compression=hyperparameters["compression"])
        if hyperparameters["model_type"] == "mlp":
            model = MethylMLP(num_classes=num_classes, num_inputs=num_inputs, num_lin_blocks=hyperparameters["num_blocks"], hidden_dim=hyperparameters["dim_hidden"])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr_cls"])
    criterion = torch.nn.CrossEntropyLoss()
    dataset = Methylation_ds()
    split = int(0.8 * len(dataset))
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset) - split], generator=torch.Generator().manual_seed(seed))
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=math.ceil(len(train_dataset)/batch_size) *hyperparameters["epochs_cls"])
    wandb.init(project="methyl_cls", config=config)
    loss_avg = 0.0
    accuracy_avg = 0.0
    with tqdm(total=math.ceil(len(train_dataset)/batch_size) *hyperparameters["epochs_cls"]) as pbar:
        model.train()
        for epoch in range(hyperparameters["epochs_cls"]):
            for _, (x, y) in enumerate(dataloader):
                start = time.time()
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = model(x, cls = True)

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

                log_dict = {}
                avg_acc = y_pred.argmax(dim=1).eq(y).sum().item() / len(y)
                log_dict["accuracy"] = avg_acc
                log_dict["loss_cls"] = loss.item()
                log_dict["time_per_step_cls"] = time.time() - start
                log_dict["lr_cls"] = optimizer.param_groups[0]["lr"]

                wandb.log(log_dict)
                loss_avg = 0.99 * loss_avg + 0.01 * loss.item()
                accuracy_avg = 0.99 * accuracy_avg + 0.01 * avg_acc
                loss_avg_corrected = loss_avg / (1 - 0.99**(pbar.n+1))
                accuracy_avg_corrected = accuracy_avg / (1 - 0.99**(pbar.n+1))
                pbar.set_description(f"Loss: {loss_avg_corrected}, Accuracy: {accuracy_avg_corrected}")#
                pbar.update(1)

            val_loss = 0.0
            val_acc = 0.0
            model.eval()
            with torch.no_grad():
                log_dict = {}
                for _, (x, y) in enumerate(val_dataloader):
                    x = x.to(device)
                    y = y.to(device)

                    y_pred = model(x, cls = True)

                    loss = criterion(y_pred, y)
                    avg_acc = y_pred.argmax(dim=1).eq(y).sum().item() / len(y)
                    val_loss += loss.item()
                    val_acc += avg_acc
            
            val_acc /= len(val_dataloader)
            val_loss /= len(val_dataloader)
            log_dict["val_loss_cls"] = val_loss 
            log_dict["val_accuracy"] = val_acc
            wandb.log(log_dict)
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

if __name__ == "__main__":
    hyperparameters = {
        "lr_cls": 1e-4,
        "epochs_cls": 100,
        "dim_hidden": 256,
        "num_blocks": 4,
        "compression": 32,
        "model_type": "transformer",
    }
    train(hyperparameters)


