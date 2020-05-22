"""
Construct a FeatureExtraction class to retrieve
'key points', 'partitions', `saliency map' of an image
in a black-box or grey-box pattern.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from scipy.stats import norm
from skimage import data, color, io


# from keras import backend as K


# Define a Feature Extraction class.
class FeatureExtraction:
    def __init__(self, pattern='black-box'):
        self.PATTERN = pattern

        # black-box parameters
        self.IMG_ENLARGE_RATIO = 1
        self.IMAGE_SIZE_BOUND = 100
        self.MAX_NUM_OF_PIXELS_PER_KEY_POINT = 1000000

        # grey-box parameters
        self.NUM_PARTITION = 10
        self.PIXEL_BOUNDS = (0, 1)
        self.NUM_OF_PIXEL_MANIPULATION = 2

    def get_key_points(self, image, num_partition=10):
        # returns an array of y coordinates of right bounds in sorted order
        key_points = []
        print('get key points')
        h, w, c = image.shape
        for i in range(self.NUM_PARTITION):
            key_points.append(int(round(w / self.NUM_PARTITION * (i + 1), 0)))
        return key_points

    # Get partitions of an image.
    def get_partitions(self, image, num_partition=10, key_points=None):
        self.NUM_PARTITION = num_partition
        h, w, c = image.shape
        partitions = {}
        if key_points == None:
            key_points = self.get_key_points(image)
        prev = 0
        for index, points in enumerate(key_points):
            for y in range(h):
                for x in range(points - prev):
                    if index in partitions.keys():
                        partitions[index].append((y, x + prev))
                    else:
                        partitions[index] = [(y, x + prev)]
            prev = points

        kps=[i+1 for i in range(len(key_points))]
        return partitions,kps

    def word_seg(self, image):
        img = copy.deepcopy(image) * 255
        partitions = {}
        # cv2.imshow('Image',image)
        # cv2.waitKey(0)
        img = np.uint8(img)
        # print(img.shape)
        # img = cv2.medianBlur(img, 5)
        result= io.imread('exp_results/32/competitive/ub/32_SECURTY_dist4.936493157538264_iter6.png', as_gray=True)


        # bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 20)
        bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -20)

        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        index = 0
        keypoints = []
        # img=(img-50)
        temp = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 15:
                x, y, w, h = cv2.boundingRect(cnt)
                # print(x,y,w,h)
                temp.append([x, w])
                y -= 1
                h += 2

                # cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 1)
        temp = np.array(temp)
        # temp=np.sort(temp,axis=0)
        temp = temp[temp[:, 0].argsort()]
        for i, v in enumerate(temp):
            if i != 0:
                a = int((v[0] + sum(temp[i - 1])) / 2)
                keypoints.append(a)
                cv2.line(img, (a,0), (a,32),(0,0,255), 1)
                cv2.line(result, (a,0), (a,32),(0,0,255), 1)

                #         ax.plot((a, a), (0, 32), 'r-')
        keypoints.append(128)

        # cv2.imshow('segmentation', img)
        # cv2.imshow('bin',bin_img)
        # cv2.imshow('result',result)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # plt.show()
        partitions,kps = self.get_partitions(image, key_points=keypoints)
        return partitions, kps