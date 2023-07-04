import os

import wandb
from tqdm import tqdm
from argparse import Namespace, ArgumentParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import torchvision.transforms as T
from torchvision.io import read_image

from model.resnet18.model import Model
from utils.SiameseDataset import SiameseDataset
from model.siamese_loss.constrastive import ConstrastiveLoss


def parse_opt() -> Namespace:
    parser = ArgumentParser()

    # train settings
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_period", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--loss_function", type=str, default="constrative")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--distance_threshold", type=float, default=10)

    # data settings
    parser.add_argument("--train_size", type=float, default=0.8, help="split train data to create val data")
    parser.add_argument("--train_csv", type=str)
    parser.add_argument("--img_dir", type=str)
    
    # save settings
    parser.add_argument("--save_path", type=str, default="./output")

    # wandb settings
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--session_name", type=str, default="SpammingDataset")
    parser.add_argument("--project_name", type=str, default="SpammingDataset")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    
    # setting model
    device = torch.device(opt.device)
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = ConstrastiveLoss(m=opt.distance_threshold)
    
    # TODO: fix optional for image size
    opt.image_size = [105, 105]
    
    data_df = pd.read_csv(opt.train_csv)

    train_dataset = SiameseDataset(opt.img_dir, data_df, opt)
    train_dataset, val_dataset = random_split(train_dataset, [opt.train_size, 1 - opt.train_size])
    
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    
    
    # set up wandb
    if opt.wandb:
        wandb.init(
            project=opt.project_name,
            config=opt,
            name=opt.session_name
        )
        wandb.watch(model, criterion=criterion)
    
    prev_loss = 10000000
    # Train
    for epoch in tqdm(range(opt.epochs)):
        losses = []
        val_losses = []
        for X1, X2, y in train_loader:
            optimizer.zero_grad()
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            embedd1, embedd2 = model(X1, X2)
            loss = criterion(embedd1, embedd2, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        print(f"Saving model to {os.path.join(opt.save_path, 'last.pt')}")
        torch.save(model.state_dict(), os.path.join(opt.save_path, "last.pt"))
        print(sum(losses) / len(losses))
        if opt.wandb:
            wandb.log({"train/train-losses": sum(losses) / len(losses), "step": epoch})
        
        if ((epoch + 1) % opt.log_period == 0) or (epoch == 0):
            with torch.no_grad():
                for X1, X2, y in train_loader:
                    X1, X2, y = X1.to(device), X2.to(device), y.to(device)
                    embedd1, embedd2 = model(X1, X2)
                    loss = criterion(embedd1, embedd2, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses)
                prev_loss = val_loss
                if val_loss < prev_loss:
                    torch.save(model.state_dict(), os.path.join(opt.save_path, "best.pt"))
                    print(f"Saving model to {os.path.join(opt.save_path, 'best.pt')}")

                if opt.wandb:
                    wandb.log({"train/val_losses": val_loss, "step": epoch})
                    
    if opt.wandb:
        wandb.finish()