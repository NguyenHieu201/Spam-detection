import os
from typing import Any, List

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import pandas as pd


class SiameseDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        data_df: pd.DataFrame,
        transform: List[torch.nn.Module] | Any = None,
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.data_df = data_df
        self.transform = transform

    def __len__(self) -> int:
        return self.data_df.shape[0]

    def __getitem__(self, index) -> Any:
        image1_path = os.path.join(self.base_dir, self.data_df.iat[index, 0])
        image2_path = os.path.join(self.base_dir, self.data_df.iat[index, 1])
        label = self.data_df.iat[index, 2]
        
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # image1 = read_image(image1_path).type(torch.float32)
        # image2 = read_image(image2_path).type(torch.float32)

        image1 = self._transform(image1)
        image2 = self._transform(image2)

        # convert pixel to [0, 1]
        image1 = image1 / 255
        image2 = image2 / 255

        return image1, image2, label

    def _transform(self, image: torch.Tensor) -> torch.Tensor:
        if self.transform is None:
            return image
        else:
            for transform in self.transform:
                image = transform(image)

            return image
