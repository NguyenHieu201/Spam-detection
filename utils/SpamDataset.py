from typing import Any, NamedTuple

import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from torchvision.io import ImageReadMode
from argparse import Namespace


class SamplePair(NamedTuple):
    sample1: str
    sample2: str
    label: int
    
class Product(NamedTuple):
    id: str
    image_path: str

class SpamDataset(Dataset):
    
    def __init__(self, opt: Namespace, products: list[Product], num_samples: int=100, positive_ratio: float=0.5) -> None:
        super().__init__()
        self.products = products
        self.num_samples = num_samples
        self.positive_ratio = positive_ratio
        self.data_pairs = self._init_dataset()
        self.positive_transformation = T.Compose([T.RandomAffine(10), T.ColorJitter(), T.GaussianBlur((3, 3))])
        self.image_size = opt.image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_pair = self.data_pairs[index]
        sample1, sample2, label = data_pair
        
        image1 = torchvision.io.read_image(sample1, mode=ImageReadMode.GRAY)
        image2 = torchvision.io.read_image(sample2, mode=ImageReadMode.GRAY)
        
        # for positive sample, do augmentation for product
        if label == 0:
            image2 = self.positive_transformation(image2)


        # resize image
        resize = T.Resize(size=self.image_size, antialias=True)
        image1 = resize(image1)
        image2 = resize(image2)
        
        return image1 / 255, image2 / 255, label

        
    def _init_dataset(self) -> list[SamplePair]:
        num_positive = int(self.positive_ratio * self.num_samples)
        num_negative = self.num_samples - num_positive
        
        # make sample pairs
        positive_pairs = self._make_positive_pairs(num_positive)
        negative_pairs = self._make_negative_pairs(num_negative)
        return positive_pairs + negative_pairs
    
    def _make_positive_pairs(self, num_positive) -> list[SamplePair]:
        random_products = random.sample(self.products, num_positive)
        product1 = random_products
        product2 = random_products
        
        positive_paris = [SamplePair(product1[i].image_path, product2[i].image_path, label=0) 
                          for i in range(num_positive)]
        return positive_paris
        
    
    def _make_negative_pairs(self, num_negative) -> list[SamplePair]:
        random_products = random.sample(self.products, num_negative * 2)
        product1 = random_products[0::2]
        product2 = random_products[1::2]
        
        negative_paris = [SamplePair(product1[i].image_path, product2[i].image_path, label=1) 
                          for i in range(num_negative)]
        
        return negative_paris