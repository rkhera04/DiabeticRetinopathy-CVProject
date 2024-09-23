import img_transform
import os 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import sys

class DRDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.images_folder = images_folder
        self.image_files = os.listdir(images_folder)
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)
    
    def __getitem__(self, index):
        if self.train:
            image_file, label = self.data.iloc[index]
        else:
            image_file, label = self.image_files[index], -1
            image_file = image_file.replace(".jpeg", "")

        image = np.array(Image.open(os.path.join(self.images_folder, image_file+".jpeg")))

        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label, image_file

if __name__ == "__main__":

    dataset = DRDataset(
        images_folder="/Users/rajkhera/CV-Project/data/train_set/train_resized/",
        path_to_csv="/Users/rajkhera/CV-Project/data/train_set/trainLabels.csv",
        transform=img_transform.val_transforms,
    )
    loader = DataLoader(
        dataset=dataset, batch_size=32, num_workers=2, shuffle=True, pin_memory=True
    )

    for x, label, file in tqdm(loader):
        print(x.shape)
        print(label.shape)
        sys.exit()