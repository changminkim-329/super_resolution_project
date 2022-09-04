# discriminator
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
import glob

def return_disc_block(input_layer,k,n,s,name):
    out = layers.Conv2D(n,k,s,padding='same',
                        name=name+'_conv')(input_layer)
    out = layers.BatchNormalization(name=name+'_bn')(out)
    out = layers.LeakyReLU(alpha=0.2,name=name+'_leaky')(out)
    return out
    
    
disc_numbers = [64,128,128,256,256,512,512]
disc_kernels = [3]*7
disc_strides = [2,1,2,1,2,1,2]


def return_discriminator(input_shape=(None,None,3)):
    model_input = layers.Input(input_shape,name='discriminator_input')
    x = layers.Conv2D(64,3,1,padding='same',name='disc_block1_conv')(model_input)
    x = layers.LeakyReLU(alpha=0.2,name='disc_block1_leaky')(x)

    for i in range(7):
        name = 'disc_block'+str(i+2)
        x = return_disc_block(x,disc_kernels[i],disc_numbers[i],
                              disc_strides[i],name)


    x = layers.Dense(1024,name='disc_block10_dense')(x)
    x = layers.LeakyReLU(alpha=0.2,name='disc_block10_leaky')(x)
    model_output = layers.Dense(1,name='discriminator_output',
                                activation='sigmoid')(x)


    disc_model = models.Model(inputs=model_input,outputs=model_output)