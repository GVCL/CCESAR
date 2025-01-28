from PIL import Image
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from constants import *
def calculate_iou(image_path, mask_path):
    image=imread(image_path).astype(bool)
    mask=imread(mask_path).astype(bool)
    image=resize(image, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
    image_array = np.array(image)
    mask_array = np.array(mask)
    intersection = np.logical_and(image_array, mask_array)
    union = np.logical_or(image_array, mask_array)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    return iou

