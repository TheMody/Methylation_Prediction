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

def train():
    if model_type == "transformer":
        model = EncoderModelPreTrain(num_classes=num_classes, num_tokens=num_inputs)
    if model_type == "mlp":
        model = MethylMLP(num_classes=num_classes, num_inputs=num_inputs)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = Methylation_ds()
    split = int(0.9 * len(dataset))
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset) - split], generator=torch.Generator().manual_seed(42))
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=math.ceil(len(train_dataset)/batch_size) *epochs)
    wandb.init(project="methyl_cls", config=config)
    loss_avg = 0.0
    accuracy_avg = 0.0
    with tqdm(total=math.ceil(len(train_dataset)/batch_size) *epochs) as pbar:
        model.train()
        for epoch in range(epochs):
            for _, (x, y) in enumerate(dataloader):
                start = time.time()
                x = x.to(device)
                y = [y_s.to(device) for y_s in y]

                optimizer.zero_grad()
                y_pred = model(x)

                acc_loss = torch.tensor(0.0).to(device)
                for i,y_s in enumerate(y):
                    loss = criterion(y_pred[i], y_s)
                    acc_loss += loss
                acc_loss.backward()
                optimizer.step()
                scheduler.step()

                log_dict = {}
                avg_acc = 0.0
                for i,y_s in enumerate(y):
                    log_dict["accuracy"+ str(i)] = y_pred[i].argmax(dim=1).eq(y_s).sum().item() / len(y_s)
                    avg_acc += log_dict["accuracy"+ str(i)]
                avg_acc = avg_acc / len(y)
                log_dict["loss"] = acc_loss.item()
                log_dict["time_per_step"] = time.time() - start
                log_dict["lr"] = optimizer.param_groups[0]["lr"]

                wandb.log(log_dict)
                loss_avg = 0.99 * loss_avg + 0.01 * acc_loss.item()
                accuracy_avg = 0.99 * accuracy_avg + 0.01 * avg_acc
                pbar.set_description(f"Loss: {loss_avg}, Accuracy: {accuracy_avg}")#
                pbar.update(1)

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                log_dict = {}
                for i in range(len(num_classes)):
                    log_dict["val_accuracy"+ str(i)] = 0.0
                for _, (x, y) in enumerate(val_dataloader):
                    x = x.to(device)
                    y = [y_s.to(device) for y_s in y]
                    y_pred = model(x)
                    acc_loss = torch.tensor(0.0).to(device)
                    for i,y_s in enumerate(y):
                        loss = criterion(y_pred[i], y_s)
                        acc_loss += loss
                    val_loss += acc_loss.item()

                    for i,y_s in enumerate(y):
                        log_dict["val_accuracy"+ str(i)] += y_pred[i].argmax(dim=1).eq(y_s).sum().item() / len(y_s)
                for i in range(len(num_classes)):
                    log_dict["val_accuracy"+ str(i)] /= len(val_dataloader)
            val_loss /= len(val_dataloader)
            log_dict["val_loss"] = val_loss 
            wandb.log(log_dict)
            avg_acc = np.mean([log_dict["val_accuracy"+ str(i)] for i in range(len(num_classes))])
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {avg_acc}")

if __name__ == "__main__":
    train()


