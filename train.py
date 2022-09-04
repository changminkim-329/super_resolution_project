#module
from generator import return_generator
from disciminator import return_discriminator
from tensorflow import keras
from IPython.display import clear_output
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2


#model
generator_model = return_generator()
disc_model = return_discriminator()

#Optimizer
learning_rate = 0.0002
beta = 0.5
gen_optimizer = keras.optimizers.Adam(learning_rate,beta)
disc_optimizer = keras.optimizers.Adam(learning_rate,beta)

#setting
STEPS = 30000
EPOCHS = 20
alpha = 1e-3


@tf.function()
def step(lr,hr_real):
    with tf.GradientTape() as gene_tape, tf.GradientTape() as disc_tape:
        hr_fake = generator_model(lr,training=True)
        disc_fake = disc_model(hr_fake,training=True)
        disc_real = disc_model(hr_real,training=True)
        
        # generator 
        get_gen_adversarial_Loss = gen_adversarial_Loss(disc_fake)
        get_gen_content_loss = content_loss(hr_real,hr_fake)
        perceptual_loss = 5*(get_gen_content_loss + alpha*get_gen_adversarial_Loss)
        # discriminator
        discriminator_loss = disc_adversarial_Loss(disc_real,disc_fake)
    disc_gradient = disc_tape.gradient(discriminator_loss,disc_model.trainable_variables)
    gene_gradient = gene_tape.gradient(perceptual_loss,generator_model.trainable_variables)
    
    disc_optimizer.apply_gradients(zip(disc_gradient,disc_model.trainable_variables))
    gen_optimizer.apply_gradients(zip(gene_gradient,generator_model.trainable_variables))
    
    return perceptual_loss, discriminator_loss
    
    
# visualization generator image 
def visual_image(lr_image,fake_image, real_image):
    plt.figure(figsize=(10,6))
    plt.subplot(1,3,1)
    plt.title('lr_image')
    plt.imshow(lr_image*0.5+0.5)
    
    plt.subplot(1,3,2)
    plt.title('fake_hr_image')
    plt.imshow(fake_image*0.5+0.5)

    plt.subplot(1,3,3)
    plt.title('hr_image')
    plt.imshow(real_image*0.5+0.5)

    plt.show()
    
# visualization loss 
def visual_plot(train_g_loss,valid_g_loss,train_d_loss,valid_d_loss):
    plt.subplot(1,2,1)
    plt.title('generator_loss')
    plt.plot(train_g_loss)
    plt.plot(valid_g_loss,'--')

    plt.subplot(1,2,2)
    plt.title('discriminator_loss')
    plt.plot(train_d_loss)
    plt.plot(valid_d_loss,'--')

    plt.show()
    

# train

def train():
    g_losses = []
    d_losses = []
    val_g_losses = []
    val_d_losses = []
    for epoch in range(EPOCHS):
        train_ds = image_generator('/jupyterNotebook/datasets/div2k/DIV2K_train_HR/*')
        for i, (lr,hr) in enumerate(train_ds):
            g_loss, d_loss = step(lr,hr)


            if (i+1) % 10 == 0:
                # train_loss
                g_losses.append(g_loss)
                d_losses.append(d_loss)


                # validation
                valid_ds = image_generator('/jupyterNotebook/datasets/div2k/DIV2K_valid_HR/*')
                lr,hr = next(iter(valid_ds))
                hr_fake = generator_model.predict(lr)
                disc_fake = disc_model.predict(hr_fake)
                disc_real = disc_model.predict(hr)

                get_gen_adversarial_Loss = gen_adversarial_Loss(disc_fake)
                get_gen_content_loss = content_loss(hr,hr_fake)
                perceptual_loss = get_gen_content_loss + alpha*get_gen_adversarial_Loss

                discriminator_loss = disc_adversarial_Loss(disc_real,disc_fake)

                val_g_losses.append(perceptual_loss)
                val_d_losses.append(discriminator_loss)

                clear_output(wait=True)
                print(f"EPOCH[{epoch}] - STEP[{i+1}] \nGenerator_loss:{g_loss:.4f} \nDiscriminator_loss:{d_loss:.4f}", end="\n\n")

                print(f'val_g_loss: {perceptual_loss:.4f}')
                print(f'val_d_loss: {discriminator_loss:.4f}')

                visual_image(lr[0],hr_fake[0],hr[0])
                visual_plot(g_losses,val_g_losses,d_losses,val_d_losses)