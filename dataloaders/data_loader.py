from torch.utils.data import Dataset
import os.path
import glob
import imageio

class Dataset_MSS_test(Dataset):
    def __init__(self, data_dir, transform=None):
        """transform：transforms for original img"""
        self.data_dir = data_dir
        self.transform = transform

        self.images_dir = glob.glob(os.path.join(self.data_dir,'*.tif'))
        self.images_name = [os.path.splitext(os.path.basename(i))[0] for i in self.images_dir]

    def __len__(self):
        return len(self.images_dir)
    def __getitem__(self, idx):
        name = self.images_name[idx]
        img = imageio.imread(self.images_dir[idx])  # HWC
        if self.transform:
            img = self.transform(img)

        return name, img

