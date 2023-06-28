import os
import glob
import pandas as pd
from torch.utils.data import DataLoader
from utils.SpamDataset import SpamDataset, Product
from argparse import Namespace


image_path = "D:\Downloads\spamimg_test\Products2"
id_products = os.listdir(image_path)
products = []

opt = Namespace(**{
    "image_size": (500, 500)
})

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

dataset = SpamDataset(opt, products, num_samples, positive_ratio)
image0, image1, label = dataset[0]
print(image0.shape)
print(image1.shape)