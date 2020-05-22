# -*- coding: utf-8 -*-
"""
Construct a NeuralNetwork class to include operations
related to various datasets and corresponding models.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""
# import keras
# from keras.datasets import mnist, cifar10
from skimage import io, color, exposure, transform
import pandas as pd
import numpy as np
from utils import utils
import h5py
import os
import glob

CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'.!?,\"&£$€:\\%/@()*+"

# Define a Neural Network class.
class DataSet:
    # Specify which dataset at initialisation.
    def __init__(self, data_set, trainOrTest):
        self.data_set = data_set

        if self.data_set== 'crnn' or self.data_set=='small':
            data_size=100
            input_height, input_width= 32,128
            train_label_path = 'crnndata/ICDAR_C1_training/gt.csv'
            train_root_dir = 'crnndata/ICDAR_C1_training'
            test_label_path = 'crnndata/ICDAR_C1_testing/gt.csv'
            test_root_dir = 'crnndata/ICDAR_C1_testing'
            utilities=utils(32,data_size,CHAR_VECTOR,input_height,input_width)
            x, y, maxlen = utilities.image2array(test_label_path, test_root_dir)


        else:
            print("Unsupported dataset %s. Try 'mnist' or 'cifar10'." % data_set)
            exit()

        self.x = x
        self.y = y

    # get dataset 
    def get_dataset(self):
        return self.x, self.y

    def get_input(self, index):
        return self.x[index]

    def preprocess_img(self, img, img_rows, img_cols):
        # Histogram normalization in y
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)

        # central scrop
        min_side = min(img.shape[:-1])
        centre = img.shape[0] // 2, img.shape[1] // 2
        img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2, :]

        # rescale to standard size
        img = transform.resize(img, (img_rows, img_cols))

        # roll color axis to axis 0
        # img = np.rollaxis(img, -1)

        return img

    def get_class(self, img_path):
        return int(img_path.split('/')[-2])
