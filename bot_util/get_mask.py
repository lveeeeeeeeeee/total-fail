import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, SpatialDropout2D, Dropout
from tensorflow.keras.losses import BinaryFocalCrossentropy

from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras import activations

from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import saving
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow import keras
from keras.saving import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

@keras.saving.register_keras_serializable('guryavkin', 'dice_coef')
def dice_coef(y_true, y_pred, smooth=100):
    """
    Custom implementation of dice coefficient.
    y_true - True Positive value.
    y_pred - Predicted value.
    return - Dice coefficient.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

@keras.saving.register_keras_serializable('guryavkin', 'dice_loss')
def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred, 0.1)
    
model_path = "everything/model.keras"
model = load_model(model_path)

def predict_on_input(image: np.ndarray):
    width, height = image.shape[0], image.shape[1]
    if max([height, width]) > 2048:
        return np.array(None)
    if min([height, width]) < 512:
        return np.array(None)
    
    image_pred = cv2.resize(image, [512, 512], interpolation=cv2.INTER_LINEAR)
    image_pred = np.expand_dims(image_pred, 0)
    predict = model.predict(image_pred).astype('uint8')
    print(predict.shape)
    predict = cv2.resize(predict[0], [height, width], interpolation=cv2.INTER_LINEAR)
    
    image[:, :, 2] *= predict
    return image


def log_too_small():
    print("too_small_image")

def log_too_big():
    print("too_large_image")

# image = cv2.imread("everything\\in.jpeg", cv2.IMREAD_COLOR)
# y_pred = predict_on_input(image)

# image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
# print(y_pred.shape, image.shape)
# cv2.imwrite("everything\\result.jpeg", image)