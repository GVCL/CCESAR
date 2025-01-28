from pyexpat import model
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.layers import Input
from constants import *


class Classification_Model:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        class_model = models.Sequential(
            [
                Input(shape=(IMG_HEIGHT_CLASSIFICATION, IMG_WIDTH_CLASSIFICATION, 3)),
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        return class_model

    def get_model(self):
        return self.model
