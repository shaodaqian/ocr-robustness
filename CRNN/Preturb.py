from Model import CRNN_model
import pandas as pd
import cv2
import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from utils import dist, utils, showImg

model_path='history/training_checkpoints/reg5750.hdf5'


class preturb:

    def __init__(self,width,height):
        self.width=width
        self.height=height

    def random(self,image,model,model_path,delta,util):
        model.load(model_path)
        output=model.predict(image)
        gt,prob=util.ctc_beam_decoder(output)
        memory=np.zeros((10,self.height,self.width))
        count=0
        for i in range(1):
            preturbed=gt
            dup=np.copy(image)
            while preturbed==gt:
                w=int(np.random.random_sample()*self.width)
                h=int(np.random.random_sample()*self.height)
                amount=delta*(np.random.random_sample()-0.5)
                dup[h][w][0]+= amount
                memory[i][h][w]+= amount
                output = model.predict(dup)
                preturbed, logp = util.ctc_beam_decoder(output)
                count+=1
                print(logp)
            print(preturbed)
            print(dist().l2Distance(image,dup))
            print(count)
        with open('preturb/f.txt', 'wb') as fp:
            pickle.dump(memory, fp)
        plt.imshow(np.squeeze(image), cmap='gray')
        plt.show()

    def word_seg(self,image):
        image=image*255
        # cv2.imshow('Image',image)
        # cv2.waitKey(0)
        image=np.uint8(image)
        print(image.shape)
        bin_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -20)
        # bin_img=np.expand_dims(bin_img,axis=2)
        print(bin_img.shape)
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        for cnt in contours:
            print(cv2.contourArea(cnt))
            if cv2.contourArea(cnt) > 15:
                x, y, w, h = cv2.boundingRect(cnt)
                print(x,y,w,h)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow('Image',bin_img)
        cv2.imshow('Image',image)
        cv2.waitKey(0)

# with open ('preturb/memory.txt', 'rb') as fp:
#     itemlist = pickle.load(fp)
# plt.figure()
# itemlist=np.squeeze(itemlist)
# print(itemlist.shape)
# showImg(itemlist)
# plt.show()