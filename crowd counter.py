#!/usr/bin/env python
# coding: utf-8

# In[9]:


from tensorflow.keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from matplotlib import cm
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as io
from PIL import Image
import PIL
import h5py
import os
import glob
import cv2
import random
import math
import sys
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, concatenate, add, UpSampling2D
from keras.initializers import RandomNormal


def CrowdNet_vgg19(input_shape=(None, None, 3)):

    input_flow = Input(shape=input_shape)
    dilated_conv_kernel_initializer = RandomNormal(stddev=0.01)

    # front-end
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_flow)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # back-end
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)

    output_flow = Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(x)
    model = Model(inputs=input_flow, outputs=output_flow)

    front_end = VGG19(weights='imagenet', include_top=False)

    weights_front_end = []
    for layer in front_end.layers:
        if 'conv' in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(front_end.layers)):
        if counter_conv >= 10:
            break
        if 'conv' in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])
            counter_conv += 1

    return model


def create_img_pred(path):
    #Function to load,normalize and return image 
    im = Image.open(path).convert('RGB')
    
    im = np.array(im)
    
    im = im/255.0
    
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225


    im = np.expand_dims(im,axis  = 0)
    return im

def predict(image):
    #Function to load image,predict heat map, generate count and return (count , image , heat map)
    model = CrowdNet_vgg19()
    model.load_weights('trainedA_100_300_vgg19_bn.hdf5')
    image = create_img_pred(image)
    ans = model.predict(image)
    count = np.sum(ans)
    plt.imshow(ans.reshape(ans.shape[1],ans.shape[2]) , cmap = cm.jet)
    plt.title(f'Estimated count: {round(count)}')
    plt.show()
    return plt
    
import streamlit as st 
from PIL import Image


st.title("LT6 Crowd Counter")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Counting...")
    plt = predict(uploaded_file)
    st.pyplot(fig=plt)

