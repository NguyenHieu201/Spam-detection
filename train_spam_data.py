import os

import wandb
from tqdm import tqdm
from argparse import Namespace, ArgumentParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import torchvision.transforms as T

from model.sample.model import SiameseNetwork
from utils.SiameseDataset import SiameseDataset
from model.siamese_loss.constrastive import ConstrastiveLoss
from utils.SpamDataset import Product, SpamDataset


def parse_opt() -> Namespace:
    parser = ArgumentParser()

    # train settings
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_period", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--loss_function", type=str, default="constrative")
    parser.add_argument("--device", type=str, default="cpu")

    # data settings
    parser.add_argument("--train_path", type=str, help="spamming data path")
    parser.add_argument("--train_size", type=float, default=0.8, help="split train data to create val data")
    
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
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = ConstrastiveLoss(m=2)
    
    # setting dataset
    image_path = opt.train_path
    id_products = os.listdir(image_path)
    products = []
    opt.image_size = [500, 500]

    for id_product in id_products:
        # print(id_product)
        # print(f"{os.path.join(image_path, id_product)}")
        image = os.listdir(os.path.join(image_path, id_product))
        for img in image:
            if img.endswith(".jpg"):
                image = img
        
                image = os.path.join(
                    image_path,
                    id_product,
                    image
                )
                product = Product(id=id_product, image_path=image)
                products.append(product)



    print(f"total samples available is: {len(products)}")
    num_samples = 100
    positive_ratio = 0.5

    train_dataset = SpamDataset(opt, products, num_samples, positive_ratio)
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
    
    prev_loss = 1e10
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
            
        torch.save(model.state_dict(), os.path.join(opt.save_path, "last.pt"))
        print(sum(losses) / len(losses))
        if opt.wandb:
            wandb.log({"train/train-losses": sum(losses) / len(losses)})
        
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
                if opt.wandb:
                    wandb.log({"train/val_losses": val_loss})
                    
    if opt.wandb:
        wandb.finish()