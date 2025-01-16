import torch
from model import MethylMLP, EncoderModelPreTrain, xtransformer

from config import *
from dataset import Methylation_ds
import wandb
from cosine_scheduler import CosineWarmupScheduler
import time
from tqdm import tqdm
import math

def pre_train(hyperparameters):
    if hyperparameters["model_type"] == "transformer":
        model = xtransformer(num_classes=num_classes, num_tokens=num_inputs, hidden_dim=hyperparameters["dim_hidden"], n_layers=hyperparameters["num_blocks"], compression=hyperparameters["compression"])
    model = model.to(device)
    uncompiled_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"])
    dataset = Methylation_ds(name = "GPL8490", interesting_values=[])
    split = int(0.95 * len(dataset))
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset) - split], generator=torch.Generator().manual_seed(seed))
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    scheduler = CosineWarmupScheduler(optimizer, warmup=300, max_iters=math.ceil(len(train_dataset)/(batch_size*gradient_accumulation_steps)) *hyperparameters["epochs"])
    wandb.init(project="methyl_cls_pretrain", config=config)
    print(hyperparameters)
    loss_avg = 0.0
    best_val_loss = 1e10
    with tqdm(total=math.ceil(len(train_dataset)/(batch_size*gradient_accumulation_steps)) *hyperparameters["epochs"]) as pbar:
        for epoch in range(hyperparameters["epochs"]):
            model.train()
            train_iter = iter(dataloader)
            for step in range(len(dataloader)//gradient_accumulation_steps):
                accloss = 0.0
                start = time.time()
                optimizer.zero_grad()

                for micro_step in range(gradient_accumulation_steps):
                    x,_ = next(train_iter)
                    x = x.to(device)
                    x_pred = model(x, mask_ratio = hyperparameters["mask_ratio"], selfmask = True)
                    loss = model.loss
                    loss = loss/ gradient_accumulation_steps
                    accloss += loss.item()
                    loss.backward()

                optimizer.step()
                scheduler.step()

                log_dict = {}
                log_dict["loss"] = accloss
                log_dict["time_per_step"] = time.time() - start
                log_dict["lr"] = optimizer.param_groups[0]["lr"]

                wandb.log(log_dict)
                loss_avg = 0.99 * loss_avg + 0.01 * accloss
                loss_avg_corrected = loss_avg / (1 - 0.99**(pbar.n+1))  
                pbar.set_description(f"Loss: {loss_avg_corrected}")
                pbar.update(1)

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                log_dict = {}
                for _, (x, _) in enumerate(val_dataloader):
                    x = x.to(device)
                    x_pred = model(x, mask_ratio = hyperparameters["mask_ratio"], selfmask = True)
                    loss = model.loss
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)
            log_dict["val_loss"] = val_loss 
            wandb.log(log_dict)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(uncompiled_model.state_dict(), folder+"/best_model.pth")
                print(f"Validation Loss: {val_loss}, Model saved")
            else:
                print(f"Validation Loss: {val_loss}")

    #load the best model saved during training
    uncompiled_model.load_state_dict(torch.load(folder+"/best_model.pth",weights_only=True))
    return uncompiled_model

if __name__ == "__main__":
    hyperparameters = {
        "lr": 8e-5,
        "dim_hidden": 256,
        "num_blocks": 4,
        "compression": 32,
        "model_type": "transformer",
        "mask_ratio": 0.15,
        "epochs": 10,
    }
    pre_train(hyperparameters)


