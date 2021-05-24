


# python drive.py /opt/p4_model/model_2021_05_17.h5

# python drive.py /opt/p4_model/model_2021_05_17.h5 run1


# python model_generator.py 


# '/opt/p4_model/model_2021_05_15.h5'

import csv
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle
from PIL import Image
from tqdm.auto import tqdm, trange
import random
import sklearn

import matplotlib.pyplot as plt

# check that all data version are correect 

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

from sklearn.model_selection import train_test_split

import preprocess as pp

from math import ceil


root_path = '/home/workspace/CarND-Behavioral-Cloning-P3/'
model_path = '/opt/p4_model/'



# combine the data from various runs 
sample_2laps  =  pp.get_samples(root_path,'data_custom_2_laps_succ/driving_log.csv')
sample_obr    =  pp.get_samples(root_path,'data_custom_obr/driving_log.csv')
sample_vozvr  =  pp.get_samples(root_path,'data_custom_vozvr/driving_log.csv')
##sample_track2 =  pp.get_samples(root_path,'data_custom_track2/driving_log.csv')

samples = sample_2laps+sample_obr+sample_vozvr#+sample_track2

print('sample_2laps : ',3*len(sample_2laps))
print('sample_obr : '  ,3*len(sample_obr))
print('sample_vozvr : '  ,3*len(sample_vozvr))
#print('sample_track2 : '  ,3*len(sample_track2))
print('sample : '  ,3*len(samples))

'''
# comment this as various sources are used in this version
with open(root_path+'data_custom/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    headings = next(reader) 
    for line in reader:
        samples.append(line)
'''


from sklearn.model_selection import train_test_split
##train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# split on train,validation and test samples
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

train_samples,validation_samples = train_test_split(samples, test_size=1 - train_ratio)
test_samples, validation_samples = train_test_split(validation_samples, test_size=test_ratio/(test_ratio + validation_ratio)) 

print(len(train_samples),len(validation_samples),len(test_samples))


print(root_path+'data_custom/driving_log.csv')



batch_size=32

# compile and train the model using the generator function
train_generator = pp.generator(train_samples, batch_size=batch_size)
validation_generator = pp.generator(validation_samples, batch_size=batch_size)
test_generator = pp.generator(test_samples, batch_size=batch_size)



##define the model architecture
model = keras.Sequential()
model.add(Lambda(lambda x: x/255-0.5,input_shape=(31,64,3)))
#model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(12,(5,5),activation='relu')) #added
model.add(Conv2D(24,(5,5),activation='relu'))
model.add(Conv2D(36,(5,5),activation='relu'))
#model.add(Dropout(0.5)) # addded dropout 2021-05-17
model.add(Conv2D(48,(5,5),activation='relu'))
model.add(Conv2D(64,(5,5),activation='relu'))
model.add(Conv2D(64,(5,5),activation='relu')) # added
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(75)) # added
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,steps_per_epoch=ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=ceil(len(validation_samples)/batch_size),
epochs=5, verbose=1)


score= model.evaluate_generator(test_generator,steps=ceil(len(test_samples)/batch_size))
print('test generator score',score)

model.save(model_path+'model_2021_05_17.h5')


'''
history_object = model.fit_generator(train_generator,steps_per_epoch =
    ceil(len(train_samples)/batch_size), validation_data = 
    validation_generator,
    validation_steps=ceil(len(validation_samples)/batch_size), 
    nb_epoch=5, verbose=1)
'''


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
##plt.show()
plt.savefig(root_path + 'Learning.png')





