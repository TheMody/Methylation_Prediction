import torch
import numpy as np
from model import MethylMLP, EncoderModelPreTrain, xtransformer

from config import *
from dataset_shizo import Methylation_ds
import wandb
from cosine_scheduler import CosineWarmupScheduler
import time
from tqdm import tqdm
import math

def lin_reg_baseline(ds, val_ds):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    x = []
    y = []
    for i in range(len(ds)):
        x.append(ds[i][0].numpy())
        y.append(ds[i][1].numpy())

    x = np.stack(x)
    y = np.stack(y)

    reg = LinearRegression()
    reg.fit(x, y)

    x = []
    y = []
    for i in range(len(val_ds)):
        x.append(val_ds[i][0].numpy())
        y.append(val_ds[i][1].numpy())

    x = np.stack(x)
    y = np.stack(y)

    y_pred = reg.predict(x)
    # print(y_pred)
    # print(np.round(y_pred))
    # print(y)
    acc = np.sum(np.round(y_pred) == y)/len(y)
    print("linear accuracy on this split", acc)

def train(hyperparameters,model = None):
    if model is None:
        if hyperparameters["model_type"] == "transformer":
            model = xtransformer(num_classes=num_classes, num_tokens=num_inputs, hidden_dim=hyperparameters["dim_hidden"], n_layers=hyperparameters["num_blocks"], compression=hyperparameters["compression"])
            # if hyperparameters["dim_hidden"] * hyperparameters["num_blocks"] >= 2048:
            #     batch_size = 16
            # if hyperparameters["dim_hidden"] * hyperparameters["num_blocks"] >= 8096:
            #     batch_size = 8
            # if hyperparameters["compression"] <= 16:
            #     batch_size = 8
        if hyperparameters["model_type"] == "mlp":
            model = MethylMLP(num_classes=num_classes, num_inputs=num_inputs, num_lin_blocks=hyperparameters["num_blocks"], hidden_dim=hyperparameters["dim_hidden"])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr_cls"])
    criterion = torch.nn.CrossEntropyLoss()
    dataset = Methylation_ds()
    split = int(0.8 * len(dataset))
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset) - split], generator=torch.Generator().manual_seed(seed))

    #lin regression baseline
    lin_reg_baseline(train_dataset, val_dataset)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=math.ceil(len(train_dataset)/batch_size) *hyperparameters["epochs_cls"])
    wandb.init(project="methyl_cls", config=config)

    print(hyperparameters)
    loss_avg = 0.0
    accuracy_avg = 0.0
    best_val_loss = 1e10
    best_val_acc = 0.0  
    with tqdm(total=math.ceil(len(train_dataset)/batch_size) *hyperparameters["epochs_cls"]) as pbar:
        model.train()
        for epoch in range(hyperparameters["epochs_cls"]):
            for _, (x, y) in enumerate(dataloader):
                start = time.time()
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = model(x, cls = True)
               # print(y_pred)

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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            log_dict["val_loss_cls"] = val_loss 
            log_dict["val_accuracy"] = val_acc
            wandb.log(log_dict)
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
    wandb.finish()

    return best_val_acc

if __name__ == "__main__":
    hyperparameters = {
        "lr_cls": 1e-4,
        "epochs_cls": 100,
        "dim_hidden": 256,
        "num_blocks": 4,
        "compression": 32,
        "model_type": "xtransformer",
    }
    train(hyperparameters)


