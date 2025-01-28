import os
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
from constants import *


def read_images(path, read_mask=True, isBuilt=False):
    ids = next(os.walk(f"{path}/images"))[2]
    X = None
    Y = None
    if read_mask:
        X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        Y = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    else:
        X = np.zeros((len(ids), IMG_HEIGHT_CLASSIFICATION, IMG_WIDTH_CLASSIFICATION, IMG_CHANNELS), dtype=np.uint8)
        Y = np.zeros((len(ids), 1), dtype=bool)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        img_path = os.path.join(path, "images", id_)
        img = imread(img_path)[:, :, :IMG_CHANNELS]
        if read_mask:
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
        else:
            img = resize(img, (IMG_HEIGHT_CLASSIFICATION, IMG_WIDTH_CLASSIFICATION), mode="constant", preserve_range=True)
            
        X[n] = img.astype(np.uint8)
        if read_mask:
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
            mask_id = id_.replace(".tiff", "_mask.tiff")
            mask_path = os.path.join(path, "masks", mask_id)
            mask_ = imread(mask_path)
            mask_ = resize(
                mask_, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True
            )
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_).astype(bool)
            Y[n] = mask
        else:
            Y[n] = isBuilt
    return (X, Y)
