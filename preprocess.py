import csv
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle
from PIL import Image
#from tqdm.auto import tqdm, trange
import random
import sklearn

# check that all data version are correect 

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

from sklearn.model_selection import train_test_split

root_path = '/home/workspace/CarND-Behavioral-Cloning-P3/'

def get_3_images(c,l,r,c_f,c_l,c_r):
    f, (ax1, ax2, ax3,ax4,ax5,ax6) = plt.subplots(1, 6, figsize=(20,10))
    ax1.imshow(c)
    ax1.set_title(' Center', fontsize=7)
    ax2.imshow(l)
    ax2.set_title(' Left',   fontsize=7)
    ax3.imshow(r)
    ax3.set_title(' Right',  fontsize=7)
    ax4.imshow(c_f)
    ax4.set_title(' Center_flipped',  fontsize=7)
    ax5.imshow(c_l)
    ax5.set_title(' Left_flipped',  fontsize=7)
    ax6.imshow(c_r)
    ax6.set_title(' Right_flipped',  fontsize=7)
    

def resize(image, new_dim):
    """
    Resize a given image according the the new dimension
    :param image:
        Source image
    :param new_dim:
        A tuple which represents the resize dimension
    :return:
        Resize image
    """
    return cv2.resize(image , new_dim)

    
def crop(image, top_percent, bottom_percent):
    """
    Crops an image according to the given parameters
    :param image: source image
    :param top_percent:
        The percentage of the original image will be cropped from the top of the image
    :param bottom_percent:
        The percentage of the original image will be cropped from the bottom of the image
    :return:
        The cropped image
    """
    #assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    #assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]
    
    
def generate_images_and_center_left_right_batch(batch_sample,correction= 0.2):
  
    '''
    function genertes images from center, left
    generate only on the batch level
    
    '''
    angles_center = float(batch_sample[3])
    angles_left   = angles_center + correction
    angles_right  = angles_center - correction

    image_center   =  crop(resize(cv2.imread('./data_custom_ALL/IMG/'+batch_sample[0].split('/')[-1]),(64,64)),0.3,0.2)
    image_left     =  crop(resize(cv2.imread('./data_custom_ALL/IMG/'+batch_sample[1].split('/')[-1]),(64,64)),0.3,0.2)
    image_right    =  crop(resize(cv2.imread('./data_custom_ALL/IMG/'+batch_sample[2].split('/')[-1]),(64,64)),0.3,0.2)
    
    ##get_3_images(image_center,image_left,image_right)

    
    return image_center,image_left,image_right,angles_center,angles_left,angles_right
         

def set_augment(X,y):
    '''
    function generates flipped images from the original dataset
    also generates opposite angles for measurements
    '''
    
    aug_X = np.fliplr(X)
    aug_y = -y
    
        
    return aug_X,aug_y


def generator(samples, batch_size=32):
    
    num_samples = len(samples)
    
    #change to numpy array              
                                    
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            #images = np.empty([batch_size*3, 160, 320, 3], dtype=np.uint8)
            #angles = np.empty([batch_size*3], dtype=np.uint8)
            
            
            #i=0
            for batch_sample in batch_samples:
                
                i_c,i_l,i_r,a_c,a_l,a_r = generate_images_and_center_left_right_batch(batch_sample)
          
                
                #i_c_flipped,a_c_flipped = set_augment(i_c,a_c)
                #i_l_flipped,a_l_flipped = set_augment(i_l,a_l)
                #i_r_flipped,a_r_flipped = set_augment(i_r,a_r)
              
            
            
                #get_3_images(i_c,i_l,i_r,i_c_flipped,i_l_flipped,i_r_flipped)
               
                
                
                # add only one option flip TODO
                
            
            
       
                # try only angles 
                images.extend([i_c,i_l,i_r])
                angles.extend([a_c,a_l,a_r])
               
                # try all angles and flip
                #images.extend([i_c,i_l,i_r,i_c_flipped,i_l_flipped,i_r_flipped])
                #angles.extend([a_c,a_l,a_r,a_c_flipped,a_l_flipped,a_r_flipped])
                
         
        
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
        
            
            #print('Batch X_train length',len(X_train))
            #print('Batch y_train length',len(y_train))
            
            yield sklearn.utils.shuffle(X_train, y_train)
            

def get_samples(root_path,file_path):
    '''
    returns samples from specific directory
    '''
    samples=[]
    with open(root_path+file_path) as csvfile:
        reader = csv.reader(csvfile)
        headings = next(reader) 
        for line in reader:
            samples.append(line)    
    return samples
    
    
    
    