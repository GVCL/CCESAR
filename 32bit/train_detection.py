import os
import numpy as np
from constants import *
import warnings
from process_input import read_images
from unet_model import UNetModel

warnings.filterwarnings("ignore")
seed = 42
np.random.seed = seed
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
X_train, Y_train = read_images(TRAIN_PATH_NATURAL)
print("Processing natural training images and masks")

unet = UNetModel(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = unet.get_model()
results = model.fit(
    X_train, Y_train, validation_split=VAL_SPLIT, batch_size=BATCH_SIZE, epochs=EPOCHS
)
model.save(os.path.join(MODEL_SAVE_DIR, "UNet32-natural.h5"))

print("Processing developed training images and masks")
X_train, Y_train = read_images(TRAIN_PATH_DEVELOPED)

unet = UNetModel(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = unet.get_model()
results = model.fit(
    X_train, Y_train, validation_split=VAL_SPLIT, batch_size=BATCH_SIZE, epochs=EPOCHS
)
model.save(os.path.join(MODEL_SAVE_DIR, "UNet32-developed.h5.h5"))
