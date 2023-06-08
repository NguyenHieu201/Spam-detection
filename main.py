import pandas as pd
from utils.SiameseDataset import SiameseDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

df = pd.read_csv("./dataset/sign-data/train_data.csv")

dataset = SiameseDataset(
    base_dir="./dataset/sign-data/train", data_df=df, transform=[T.Grayscale()]
)
