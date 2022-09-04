import numpy as np
from skimage.measure import block_reduce
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import glob

def imread(path):
    load_img = cv2.imread(path)
    load_img = cv2.cvtColor(load_img,cv2.COLOR_BGR2RGB)
    return load_img

# 이미지 사이즈 변경
def change_size(img,resize,interpolation='linear'):
    """
    resize = (None,None)
    """
    if interpolation == 'linear':
        resize_img = tf.image.resize(img, resize,
                                     method='bicubic')
        
    if interpolation == 'cubic':
        resize_img = tf.image.resize(img, resize,
                                     method='bilinear')
        
    if interpolation == 'area':
        resize_img = tf.image.resize(img, resize,
                                     method='area')
    return resize_img

def repeat_size(img,size):
    shape = img.shape
    height = shape[0]
    width = shape[1]
    
    while True:
        resized_height = int(height*(2/3))
        resized_width = int(width*(2/3))
        if (size < height) and (size < width):
            img = cv2.resize(img,(resized_height,resized_width))
            height = resized_height
            width = resized_width
            
        else:
            img = cv2.resize(img,(size,size))
            break
            
    return img

#블러처리
def make_gausian_blur(img,sigma):
    """
    sigma: 표준편차
    """
    
    blur_img = tfa.image.gaussian_filter2d(img, (5,5),sigma)
    # cv2_imshow(blur_img)
    return blur_img

#가우시안 노이즈 처리
def make_noise(img,std,filter_type='gaussian'):
    #img = img.astype('float32')
    
    
    if filter_type == 'gaussian':     
        set_noise = std*np.random.normal(size=img.shape)
        noise_img = img + set_noise
        
    if filter_type == "poisson":
        set_noise = std*np.random.normal(size=img.shape)
        noise_img = img + set_noise
        
    return noise_img
    

# 두번 다운 샘플 후 업 샘플 다운 샘플
def down2x_up_down_resize(img,interpolation='linear',random=False):
    #img = img.astype('float32')
    shape = img.shape
    try:
        height = shape[0]//1
        width = shape[1]//1
        
    except Exception as e:
        height = 256
        width = 256
    
    down_image = img
    for i in range(2):
        shape = down_image.shape
        height = height//2
        width = width//2
        
        if random:
            interpolation=np.random.choice(interpolation_list)
        
        down_image = change_size(down_image,(width,height),
                                 interpolation)
        
    if random:
        interpolation=np.random.choice(interpolation_list)
        
    up_image = change_size(down_image,(width*2,height*2),interpolation)
    
    if random:
        interpolation=np.random.choice(interpolation_list)
            
    down_image = change_size(up_image,(width,height),
                             interpolation)
    
    resized_image = tf.clip_by_value(down_image,0,255)
    return resized_image
    
# 다운 샘플 업 샘플후 2번 다운 샘플
def down_up_down2x_resize(img,interpolation='linear',random=False):
    #img = img.astype('float32')
    shape = img.shape
    try:
        height = shape[0]//2
        width = shape[1]//2
        
    except Exception as e:
        height = 256
        width = 256
    
    
    if random:
        interpolation=np.random.choice(interpolation_list)
    resized_image = change_size(img,(width,height),interpolation) #down
    
    if random:
        interpolation=np.random.choice(interpolation_list)
    resized_image = change_size(resized_image,(width*2,height*2),interpolation)#up
    
    # down 2x
    for i in range(2):
        shape = resized_image.shape
        height = shape[0]//2
        width = shape[1]//2
        
        if random:
            interpolation=np.random.choice(interpolation_list)
        resized_image = change_size(resized_image,(width,height),interpolation)
        
    resized_image = tf.clip_by_value(resized_image,0,255)
    return resized_image
    
# 다운 + maxpooling
def down_pool_resize(img,interpolation='linear',random=False):
    #img = img.astype('float32')
    shape = img.shape
    try:
        height = shape[0]//2
        width = shape[1]//2
        
    except Exception as e:
        height = 256
        width = 256
    
    if random:
        interpolation=np.random.choice(interpolation_list)
    resized_image = change_size(img,(width,height),interpolation) #down
    
    resized_image = tf.squeeze(tf.nn.max_pool2d(resized_image[tf.newaxis],(2,2),(2,2),'VALID'))
    #block_reduce(resized_image,block_size=(2,2,1),func=np.max) #max_pool
    
    resized_image = tf.clip_by_value(resized_image,0,255)
    return resized_image
    
    
# 다운 + blur
def down_blur_resize(img,sigma=2,interpolation='linear',blured_1=True,blured_2=True,random=False):    
    #img = img.astype('float32')
    
    if(blured_1):
        if random:
            sigma = np.random.randint(1,11)
        resized_image = make_gausian_blur(img,sigma) # blur
    else:
        resized_image = img
        
    shape = resized_image.shape
    try:
        height = shape[0]//2
        width = shape[1]//2
        
    except Exception as e:
        height = 256
        width = 256
        
    if random:
        interpolation=np.random.choice(interpolation_list)
    resized_image = change_size(resized_image,(width,height),interpolation) #down
    
    if(blured_2):
        if random:
            sigma = np.random.randint(5,15)
        resized_image = make_gausian_blur(resized_image,sigma) # blur
        
    if random:
        interpolation=np.random.choice(interpolation_list)
    resized_image = change_size(resized_image,(width//2,height//2),interpolation) #down
    
    resized_image = tf.clip_by_value(resized_image,0,255)
    return resized_image
    
# 다운 + 노이즈
def down_noise_resize(img,std=5,interpolation='linear',
                      filter_type='gaussian',add_noise1=True,add_noise2=True,random=False):
    #img = img.astype('float32')
    shape = img.shape
    try:
        height = shape[0]//2
        width = shape[1]//2
        
    except Exception as e:
        height = 256
        width = 256
    
    if random:
        interpolation=np.random.choice(interpolation_list)
    resized_image = change_size(img,(width,height),interpolation) #down
    
    if add_noise1:
        if random:
            filter_type=np.random.choice(filter_list)
            std = np.random.randint(5,15)
        resized_image = make_noise(resized_image,std,filter_type) # noise
    
    if random:
        interpolation=np.random.choice(interpolation_list)
    resized_image = change_size(resized_image,(width//2,height//2),interpolation) #down
    
    if add_noise2:
        if random:
            std = np.random.randint(5,15)
            filter_type=np.random.choice(filter_list)
        resized_image = make_noise(resized_image,std,filter_type) # noise
    
    resized_image =tf.clip_by_value(resized_image,0,255)
    return resized_image
    
# 논문용 다운 샘플
def multiple_filer_size(img,sigma=2,std=5,interpolation='linear',filter_type='gaussian',
                        blured_1=True,blured_2=True,
                        add_noise1=True,add_noise2=True,random=False):
    #img = img.astype('float32')
    
    if blured_1:
        if random:
            sigma = np.random.randint(1,11)
        resized_image = make_gausian_blur(img,sigma) # blur
    else:
        resized_image = img
        
    shape = resized_image.shape
    try:
        height = shape[0]//2
        width = shape[1]//2
        
    except Exception as e:
        height = 256
        width = 256
        
    
    if random:
        interpolation=np.random.choice(interpolation_list)
    resized_image = change_size(resized_image,(width,height),interpolation) #down
    
    if add_noise1:
        if random:
            filter_type=np.random.choice(filter_list)
            std = np.random.randint(5,15)
        resized_image = make_noise(resized_image,std,filter_type) # noise
    
    if blured_2:
        if random:
            sigma = np.random.randint(5,15)
        resized_image = make_gausian_blur(img,sigma) # blur
    
    if random:
        interpolation=np.random.choice(interpolation_list)
        std = np.random.randint(5,15)
    resized_image = change_size(resized_image,(width//2,height//2),interpolation) #down
    
    if add_noise2:
        if random:
            filter_type=np.random.choice(filter_list)
        resized_image = make_noise(resized_image,std,filter_type) # noise
    
    resized_image = tf.clip_by_value(resized_image,0,255)
    return resized_image
    
interpolation_list =  ['linear','cubic','area']
filter_list = ['gaussian',"poisson"]


# 이미지 프로세싱
def preprocessing(hr):
    shape = hr.shape
    height = shape[0]
    width = shape[1]
    size = [32*4,48*4,64*4,96*4,128*4]
    size = np.random.choice(size)
    
    if height == None:
        position_y = np.random.randint(0,100)
        position_x = np.random.randint(0,100)
        
    else:
        try:
            position_y = np.random.randint(0,height-size)
            position_x = np.random.randint(0,width-size)
        except Exception as e:
            position_y = np.random.randint(0,100)
            position_x = np.random.randint(0,100)
    
    
    hr_patch = hr[position_y:position_y+size,position_x:position_x+size,:]
    
    
    
    #hr_patch = tf.image.random_crop(hr, size=size)
    
    down_sampling_func_list = [down2x_up_down_resize, down_up_down2x_resize, 
                           down_pool_resize, down_blur_resize, down_noise_resize,
                           multiple_filer_size]
    
    num = np.random.randint(0,len(down_sampling_func_list))
    
    down_sampling_func = down_sampling_func_list[num]
    
    if num == 3:
        random_choice = [(True,True),(True,False),(False,True)]
        random_choice = random_choice[np.random.randint(0,3)]
        lr_patch = down_sampling_func(hr_patch,random=True,blured_1=random_choice[0],blured_2=random_choice[1])
        
    elif num == 4:
        random_choice = [(True,True),(True,False),(False,True)]
        random_choice = random_choice[np.random.randint(0,3)]
        lr_patch = down_sampling_func(hr_patch,random=True,add_noise1=random_choice[0],add_noise2=random_choice[1])
    else:
        lr_patch = down_sampling_func(hr_patch,random=True)
    
    lr_patch = tf.cast(lr_patch,'float32')
    hr_patch = tf.cast(hr_patch,'float32')
    return lr_patch/127.5-1, hr_patch/127.5 -1
    
    
    
# 이미지 제너레이터
def image_generator(path):
    """
    이미지 제너레이터
    path: 이미지 데이터 디렉토리 경로
    """
    image_list = glob.glob(path)
    np.random.shuffle(image_list)
    
    for path in image_list:
        for i in range(6):
            img = imread(path)

            lr_image, hr_image = preprocessing(img)

            yield (lr_image[tf.newaxis], hr_image[tf.newaxis])