from typing import List
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, image: List):
        super().__init__()
        self._images = image

    def __getitem__(self, index):
        return self._images[index]

    def __len__(self):
        return len(self._images)
