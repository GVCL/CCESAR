from calendar import EPOCH
import os
import warnings
from tensorflow.keras.optimizers import Adam
import numpy as np
from classification_model import Classification_Model
from process_input import read_images
from constants import *
import numpy as np

warnings.filterwarnings("ignore")


train_folder_natural = TRAIN_PATH_NATURAL
train_folder_developed = TRAIN_PATH_DEVELOPED
print("Processing natural training images")
X_train_natural, Y_train_natural = read_images(
    train_folder_natural, read_mask=False, isBuilt=False
)

print("Processing developed training images")
X_train_developed, Y_train_developed = read_images(
    train_folder_developed, read_mask=False, isBuilt=True
)

X_train = np.concatenate((X_train_natural, X_train_developed), axis=0)
Y_train = np.concatenate((Y_train_natural, Y_train_developed), axis=0)


indices = np.random.permutation(len(X_train))
X_train = X_train[indices]
Y_train = Y_train[indices]

print(X_train.shape)
print(Y_train.shape)

classification_model = Classification_Model()
model = classification_model.get_model()
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=1e-5),
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VAL_SPLIT,
)


model.save(os.path.join(MODEL_SAVE_DIR,'32bit-classification.h5'))
