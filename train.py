import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from matplotlib import pyplot as plt
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

def get_transforms(params: Dict[str, Any]) -> transforms.Compose:
    """Return a torchvision transform pipeline with ToTensor and normalization."""
    mean, std = params["mean"], params["std"]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    

def get_loaders(params: Dict[str, Any])-> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders for the MNIST dataset."""
    
    train_tf = get_transforms(params)

    full_train = datasets.MNIST(params["data_dir"],train=True,download=True,transform=train_tf)
    generator = torch.Generator().manual_seed(params["seed"])
    train_size = int(0.83 * len(full_train))   
    val_size   = len(full_train) - train_size 
    train_ds, val_ds = random_split(full_train, [train_size, val_size],generator=generator)

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],shuffle=True,  num_workers=params["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],shuffle=False, num_workers=params["num_workers"])

    return train_loader, val_loader



def plot_losses(train_losses: List[float], val_losses: List[float],save_path: Optional[str] = None) -> None:
    """Plot training and validation losses and optionally save to a file."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def create_output_dir(base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def train_one_epoch(model: nn.Module, loader: DataLoader,optimizer: torch.optim.Optimizer,criterion: nn.Module,device: torch.device,log_interval: int)-> Tuple[float, float]:
    """Train the model for one epoch and return average loss and accuracy."""
    
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(model: nn.Module,loader: DataLoader,criterion: nn.Module,device: torch.device) -> Tuple[float, float]:
    """Evaluate the model on a validation set and return average loss and accuracy."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def run_training(model: nn.Module,params: Dict[str, Any],device: torch.device) -> None:
    """Train the model with early stopping, save best weights, and log results."""
    
    run_dir = create_output_dir()
    if params["save_path"] is None:
        params["save_path"] = os.path.join(run_dir, "best_model.pth") 

    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=params["learning_rate"],weight_decay=params["weight_decay"])

    scheduler = None
    if params["scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

    best_acc     = 0.0
    best_weights = None

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, params["epochs"] + 1):
        print(f"\nEpoch {epoch}/{params['epochs']}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, params["log_interval"])
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_weights = copy.deepcopy(model.state_dict())
            best_acc = val_acc
            torch.save({"model_state_dict": best_weights, "params": params}, params["save_path"])
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if params["early_stop"] and epochs_no_improve >= params["patience"]:
            print(f"Early stopping triggered at epoch {epoch}")
            break


    if best_weights is not None:
        model.load_state_dict(best_weights)
 
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")

    plot_losses(train_losses, val_losses,save_path=os.path.join(run_dir, "loss_plot.png"))
   
    with open(os.path.join(run_dir, "train_log.txt"), "w") as f:
        f.write(f"Params: {params}\n")
        f.write(f"Best val acc: {best_acc:.4f}\n")
        f.write(f"Train losses: {train_losses}\n")
        f.write(f"Val losses: {val_losses}\n")


