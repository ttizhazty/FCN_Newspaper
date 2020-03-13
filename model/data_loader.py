import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class NewspaperDataset(Dataset):
    """
    A customized Dataset with function __len__ and __getitem__.
    """
    def __init__(self, data_dir = 'data/FCN_dataset', mode = 'train'):
        """
        Store the filenames of png to use. Specifies the mode to apply on image data processing

        Args:
            data_dir(str): directory contains the dataset
            mode(str): define the mode of the network, a str in `train`, `test`and `val`
        """
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.data_dir = data_dir
        self.image_path = os.path.join(data_dir, 'image')
        self.mask_path = os.path.join(data_dir, 'mask')
        self.filenames = os.listdir(self.image_path)
        self.image_list = [os.path.join(self.image_path, f) for f in self.filenames]

        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        self.mask_transformer = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (Tensor) corresponding mask of image
        """
        if self.mode in ['train', 'val']:
            image_name = self.image_list[index].split('/')[-1].split('.')[0]
            image = Image.open(self.image_list[index]).convert('1')

            mask = np.array(Image.open(os.path.join(self.mask_path, image_name + '_m.png')).convert('1'))
            mask = mask.reshape((256, 256, -1)).astype(np.uint8)

            image = self.image_transformer(image)
            mask = self.mask_transformer(mask) * 255
            return image, mask

        else:
            image = Image.open(image_list[index])
            image = self.image_transformer(image)
            path = self.image_list[index]
            return image, path

def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'test', 'val']:
        if split in types:
            path = os.path.join(data_dir, split)

        if split in ['train', 'val']:
            dl = DataLoader(NewspaperDataset(path, 'train'), 
                            batch_size = params.batch_size,
                            shuffle = True,
                            num_workers = params.num_workers)
        else:
            dl = DataLoader(NewspaperDataset(path, 'test'), 
                            batch_size = params.batch_size,
                            shuffle = False,
                            num_workers = params.num_workers)
        
        dataloaders[split] = dl

    return dataloaders