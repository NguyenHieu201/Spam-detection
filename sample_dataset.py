import os

from tqdm import tqdm
from argparse import Namespace, ArgumentParser
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import torchvision.transforms as T
import torchvision
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np

from model.sample.model import SiameseNetwork
from utils.SiameseDataset import SiameseDataset
from model.siamese_loss.constrastive import ConstrastiveLoss
from utils.SpamDataset import Product, SpamDataset
from model.resnet18.model import Model
from utils.SiameseDataset import SiameseDataset
import pandas as pd

import pickle as pk


def parse_opt() -> Namespace:
    parser = ArgumentParser()

    # train settings
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weight", type=str)
    parser.add_argument("--batch_size", type=int)

    # data settings
    parser.add_argument("--test_csv", type=str)
    parser.add_argument("--img_dir", type=str)

    # save settings
    parser.add_argument("--save_path", type=str, default="./output")

    # wandb settings

    opt = parser.parse_args()
    return opt


def imshow(img, text=None, should_save=False, figname=""):
    plt.clf()
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f"./output/{figname}.jpg")


if __name__ == "__main__":
    opt = parse_opt()

    # setting model
    device = torch.device(opt.device)
    model = Model().to(device)
    model.load_state_dict(torch.load(opt.weight))
    optimizer = optim.Adam(model.parameters())
    criterion = ConstrastiveLoss(m=2)

    # TODO: fix optional for image size
    opt.image_size = [105, 105]

    data_df = pd.read_csv(opt.test_csv)

    test_dataset = SiameseDataset(opt.img_dir, data_df, opt)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    dist_metrics = nn.PairwiseDistance(p=2, keepdim=True)
    threshold = 10

    results = []

    for batch_data in tqdm(test_loader):
        image1, image2, label = batch_data
        image1, image2 = image1.to(device), image2.to(device)

        # compute distance
        embedd1, embedd2 = model(image1, image2)
        distance = dist_metrics(embedd1, embedd2).detach().cpu()
        results.append(distance)

    df = pd.DataFrame(results)
    df.to_csv("/content/results.csv", index=False)
