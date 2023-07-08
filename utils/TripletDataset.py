import os

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T


class TripletDataset(Dataset):
    def __init__(self, image_dir, data_df, opt):
        super(TripletDataset, self).__init__()
        self.data_df = data_df
        self.image_dir = image_dir
        self.transform = T.Compose([T.Resize(size=(105, 105), antialias=True)])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.data_df.iat[idx, 0])
        label = self.data_df.iat[idx, 1]

        image = read_image(image_path, ImageReadMode.RGB)
        image = self.transform(image)
        if image.max() > 1:
            image = image / 255
        return image, label