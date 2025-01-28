import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
from constants import *
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
from keras.models import load_model
import os
from iou import calculate_iou
import warnings
import random
import numpy as np
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK']="True"
def load_and_preprocess_image(img_path):
    img = imread(img_path)[:, :, :IMG_CHANNELS]
    img = img * (2**32 - 1)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)
    img_array = np.array(img)
    img_array = img_array.astype(np.uint32)
    return img_array


def predict_image(model, image_path):
    img_array = load_and_preprocess_image(image_path)
    img_array = np.array([img_array])
    predictions = model.predict(img_array)
    predicted_class = int(predictions[0][0] > 0.5)
    label = "natural" if predicted_class == 0 else "built"
    return label

os.makedirs(SAVE_PATH)
start_time = time.time()

model_natural = load_model(os.path.join(MODEL_SAVE_DIR,"UNet32-natural.h5"))
model_built = load_model(os.path.join(MODEL_SAVE_DIR,"UNet32-developed.h5"))
classification_model = model = keras.models.load_model(
   os.path.join(MODEL_SAVE_DIR,"32bit-classification.h5")
)
image_dir = os.path.join(TEST_PATH,'images')
gt_dir = os.path.join(TEST_PATH,'masks')
count=0
for filename in os.listdir(image_dir):
    if filename.endswith(".tiff"):
        count+=1
        image_path = os.path.join(image_dir, filename)
        model = model_built
        if predict_image(classification_model, image_path) == "natural":
            model = model_natural
        else:
            model = model_built
        img = imread(image_path)[:, :, :IMG_CHANNELS] * (2**32 - 1)
        original_size = img.shape[:2]
        img_resized = resize(
            img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True
        )
        img_resized = np.array([img_resized]).astype(np.uint32)
        preds_test = model.predict(img_resized, verbose=1)
        preds_test_t = (preds_test > 0.5).astype(bool)
        image_num = int(filename.split(".")[0])
        save_path = os.path.join(SAVE_PATH,f"{image_num}_pred.tiff")
        imsave(save_path, preds_test_t[0])
end_time = time.time()
elapsed_time = end_time - start_time

print(f'Time elapsed : {elapsed_time} for {count} images')

ious = []
for filename in os.listdir(image_dir):
    if filename.endswith('.tiff'):
        image_num = int(filename.split('.')[0])  # Get image number
        print(image_num)
        image_path = os.path.join(gt_dir,f'{image_num}_mask.tiff')
        mask_path = os.path.join(SAVE_PATH,f'{image_num}_pred.tiff')
        iou = calculate_iou(image_path, mask_path)
        ious.append(iou)
        print(f"Image {image_num}: IoU = {iou}")

if len(ious) > 0:
    average_iou = sum(ious) / len(ious)
    print(f"Average IoU = {average_iou}")
else:
    print("No images found or IoU couldn't be calculated.") 
