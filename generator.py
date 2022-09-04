#Generator
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
import glob

def return_generator(input_shape=(None,None,3)):
    block_num = 15
    skip_connection_num = 4
    model_input = layers.Input(input_shape,name='generator_input')
    block1_conv = layers.Conv2D(64,9,1,padding="same",
                                name='block1_conv'
                               )(model_input)
    block1_prelu = layers.PReLU(shared_axes=[1,2],
                                name='block1_prelu')(block1_conv)
    
    residual_block_list = []
#     block2_residual = return_residual_block(block1_prelu,'block2')
    for i in range(15):
        if i == 0:
            x = layers.Conv2D(64,3,1,padding='same',name='block'+'_conv_conn'
                         )(block1_prelu)
            x = return_residual_block(x,'block'+str(i+2))
            residual_block_list.append(x)
        
        else:
                
            x = return_residual_block(list(residual_block_list),'block'+str(i+2))
            residual_block_list.insert(0,x)
            
            ## skip connection number : 4
            if len(residual_block_list) > skip_connection_num:
                residual_block_list.pop()
    
    x = layers.Conv2D(64,3,1,padding='same',
                                name='block'+str(block_num+2)+'_conv')(x)

    x = layers.BatchNormalization(name='block'+str(block_num+2)+'_bn')(x)
    filter_x = layers.Add(name='block'+str(block_num+2)+'_sum')([x,block1_prelu])
    
    pixel_x2 = return_pixel__shuffler_layer(filter_x,'block'+str(block_num+3))
    trans_x2 = return_conv__trans_layer(filter_x,'block'+str(block_num+3))
    
    pixel_x4 = return_pixel__shuffler_layer([pixel_x2,trans_x2],'block'+str(block_num+4))
    trans_x4 = return_conv__trans_layer([trans_x2,pixel_x2],'block'+str(block_num+4))
    
#     trans_x = return_conv__trans_layer([trans_x2,pixel_x],'block'+str(block_num+5))
    #### up_pooling
    x = return_pixel__shuffler_layer([pixel_x4,trans_x4],'block'+str(block_num+5))
    x = layers.MaxPool2D((2,2),name='block'+str(block_num+5)+'max_pool')(x)
    
    model_output = layers.Conv2D(3,9,1,padding='same',
                                 name='generator_output')(x)

    #model_output = layers.Activation(tf.tanh,name='tahh')(model_output)
    
    generator_model = models.Model(inputs=model_input, outputs=model_output)
    
    return generator_model
    
    
def return_residual_block(input_layer,name):
    x = layers.Conv2D(64,3,1,padding='same',name=name+'_conv1'
                     )(input_layer)
    x = layers.BatchNormalization(name=name+'_bn1')(x)
    x = layers.PReLU(shared_axes=[1,2],name=name+'_prelu')(x)
    
    x = layers.Conv2D(64,3,1,padding='same',name=name+'_conv2'
                     )(x)
    x = layers.BatchNormalization(name=name+'_bn2')(x)
    out = layers.Add(name=name+'_add')([x,input_layer])
    
    #out = layers.Lambda(lambda x:tf.reduce_sum(x,axis=3,keepdims=True))(x)
    
    return out
    
def return_residual_block(input_layer,name):
    if type(input_layer) == type([]):
        x = layers.BatchNormalization(name=name+'_bn1')(input_layer[0])
    else:
        x = layers.BatchNormalization(name=name+'_bn1')(input_layer)
        
        
    x = layers.PReLU(shared_axes=[1,2],name=name+'_prelu')(x)

    x = layers.Conv2D(64,3,1,padding='same',name=name+'_conv1'
                     )(x)
    x = layers.BatchNormalization(name=name+'_bn2')(x)
    
    """
    if type(input_layer) == type([]):
        input_layer.insert(0,x)
    else:
        input_layer = [input_layer]
        input_layer.insert(0,x)
    """
    
    if type(input_layer) == type([]):
        input_layer.insert(0,x)
        
        if len(input_layer) <= 2:
            input_layer.insert(0,x)
            out = layers.Add(name=name+'_add')(input_layer)
        
        else:
            input_layer.insert(0,x)
            out = layers.Add(name=name+'_add')(input_layer[:-1])
            out = layers.Concatenate(axis=-1,name=name+'concat')([out,input_layer[-1]])

        out = layers.Conv2D(64,3,1,padding='same',name=name+'_conv2')(out)
    else:
        out = layers.Add(name=name+'_add')([x,input_layer])
        out = layers.Conv2D(64,3,1,padding='same',name=name+'_conv2')(out)
    
    return out
    
def return_pixel__shuffler_layer(input_layer,name):
    if type(input_layer) == type([]):
        input_layer = layers.concatenate(input_layer)
        
    out = layers.Conv2D(256,3,1,padding='same',
                        name=name+'_conv')(input_layer)
    out = layers.Lambda(lambda x:tf.nn.depth_to_space(x,2),
                        name=name+'_pixel_shuffler')(out)
    out = layers.PReLU(shared_axes=[1,2],
                       name=name+'_prelu')(out)
    return out
    
    
def return_conv__trans_layer(input_layer,name):
    if type(input_layer) == type([]):
        input_layer = layers.concatenate(input_layer)
        
    out = layers.Conv2DTranspose(64,2,2,
                        name=name+'_conv_trans')(input_layer)
    out = layers.PReLU(shared_axes=[1,2],
                       name=name+'_prelu_trans')(out)
    return out