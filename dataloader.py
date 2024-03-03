import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils import *

class MyDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, image_transform=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.image_transform = image_transform

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name_image = annotation_line.split()[0]
        image = Image.open(name_image)

        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))
        return image
    
    def __len__(self):
        return self.length

