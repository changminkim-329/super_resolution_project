#loss
import tensorflow as tf
from tensorflow import keras

bce = keras.losses.BinaryCrossentropy()
mse = keras.losses.MeanSquaredError()

def disc_adversarial_Loss(disc_real, disc_fake):
    real_disc_loss = bce(tf.ones_like(disc_real),disc_real)
    fake_disc_loss = bce(tf.zeros_like(disc_fake),disc_fake)
    loss = real_disc_loss + fake_disc_loss
    
    return loss
    
def gen_adversarial_Loss(disc_fake):
    loss = bce(tf.ones_like(disc_fake),disc_fake)
    
    return loss
    
def content_loss(hr_real, hr_fake,feature_expractor):
    # VGG scaled with 1/12.75 as in paper
    
    
    hr_real_feature = feature_expractor(hr_real)/12.75
    hr_fake_feature = feature_expractor(hr_fake)/12.75
    
    loss = mse(hr_real_feature, hr_fake_feature)
    return loss