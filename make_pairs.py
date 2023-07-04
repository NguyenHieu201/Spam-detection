import os
from argparse import Namespace, ArgumentParser

import random

import pandas as pd

def parse_opt() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--num_sample", type=int, default=1000)
    parser.add_argument("--positive_ratio", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="./pair.csv")
    parser.add_argument("--seed", type=int, default=0)
    
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    
    random.seed(opt.seed)
    
    image_dir = opt.image_dir
    image_pair = []
    
    num_positive = int(opt.num_sample * opt.positive_ratio)
    num_negative = opt.num_sample - num_positive
    
    products = os.listdir(image_dir)
    
    # make positive
    positive_sample = random.sample(products, k=num_positive)
    for image1 in positive_sample:
        product = ""
        review = ""
        for file in os.listdir(os.path.join(image_dir, image1)):
            if ".jpg" in file or ".png" in file:
                product = file
            elif os.path.isdir(os.path.join(image_dir, image1, file)):
                reviews = [review for review in os.listdir(os.path.join(image_dir, image1, file))
                           if (".jpg" in review or ".png" in review)]
                review = reviews[0]
                review = os.path.join(file, review)
        img_path = os.path.join(image1, product)
        review_path = os.path.join(image1, review)
        
        if product == "":
            continue
        image_pair.append({
            "image1": img_path,
            "image2": review_path,
            "label": 0
        })
        
        
    
    # make negative
    negative_sample = random.sample(products, k=(2 * num_negative))
    image1 = negative_sample[0::2]
    image2 = negative_sample[1::2]
    
    for i in range(len(image1)):
        image1_path = [img for img in os.listdir(os.path.join(image_dir, image1[i]))
                       if (".jpg" in img or ".png" in img)]
        
        image2_path = [img for img in os.listdir(os.path.join(image_dir, image2[i]))
                       if (".jpg" in img or ".png" in img)]
        
        try:
            image_pair.append({
                "image1": os.path.join(image1[i], image1_path[0]),
                "image2": os.path.join(image2[i], image2_path[0]),
                "label": 1
            })
        # TODO: process the case that no image in folder
        except IndexError:
            continue
        
    df = pd.DataFrame(image_pair)
    df.to_csv(opt.output, index=False)
    
    