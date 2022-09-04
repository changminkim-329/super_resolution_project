import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
import glob
from tensorflow.keras.applications import VGG19

def get_feature_extractor(input_shape=(None,None,3)):
    # VGG scaled with 1/12.75 as in paper
    vgg = VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    print(vgg.layers[9].name)
    # 아래 vgg.layers[20]은 vgg 내의 마지막 conv layer: block5_conv4
    return models.Model(vgg.input, vgg.layers[9].output)