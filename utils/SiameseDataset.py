import os
from typing import Any, List

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T
from PIL import Image
import pandas as pd


class SiameseDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        data_df: pd.DataFrame,
        opt
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.data_df = data_df
        
        # center crop and random augmentation
        self.transform = T.Compose([ 
                                    T.CenterCrop((105, 105)),
                                    T.Resize((224, 224), antialias=True),
                                    T.ToTensor()
                                    ])

    def __len__(self) -> int:
        return self.data_df.shape[0]

    def __getitem__(self, index) -> Any:
        image1_path = os.path.join(self.base_dir, self.data_df.iat[index, 0])
        image2_path = os.path.join(self.base_dir, self.data_df.iat[index, 1])
        label = self.data_df.iat[index, 2]
        
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return image1, image2, label
