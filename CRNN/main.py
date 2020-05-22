from Model import CRNN_model,small_model, smaller_model
import pandas as pd

import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from skimage import data, color, io

from utils import utils
from Preturb import preturb
from FeatureExtraction import *

CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'.!?,\"&£$€:\\%/@()*+"


if __name__ == "__main__":

    train_label_path = 'data/ICDAR_C1_training/gt.csv'
    train_root_dir = 'data/ICDAR_C1_training'
    test_label_path = 'data/ICDAR_C1_testing/gt.csv'
    test_root_dir = 'data/ICDAR_C1_testing'
    input_height=32
    input_width=128
    data_size=None  #number or None
    val_split=0.1467


    class_num = len(CHAR_VECTOR)+1
    # print(class_num)

    epochs = 4000
    batch_size=32
    dropout=0.2
    size='small'
    mode='predict'
    new_train = False
    best_model='history/checkpoints/7636.hdf5'
    small='history/small_checkpoints/256rnn71.hdf5'
    smaller='history/weights.2500-5.82.hdf5'
    cnnout=False

    if mode=='predict':
        data_size=1
    if mode=='preturb':
        data_size=5

    utils=utils(batch_size,data_size,CHAR_VECTOR,input_height,input_width)
    if (mode!='test'):
        train_data,train_label, maxlen,text= utils.image2array(train_label_path, train_root_dir)
        print(maxlen)
        val=(train_data[128:],train_label[128:])
        # train_data=train_data[:128]
        # train_label=train_label[:128]

        print('Train data shape',train_data.shape)
        print('Train Label shape',train_label.shape)

    if mode=='test':
        test_data, test_label, maxlen, text = utils.image2array(test_label_path, test_root_dir)
        print(maxlen)
        print('Test data shape',test_data.shape)
        print('Test Label shape',test_label.shape)


    # train_label=tf.convert_to_tensor(train_label, dtype=tf.int32)
    ada=tf.keras.optimizers.Adam(learning_rate=0.0001)
    if size=='small':
        model = small_model(class_num,dropout,ada, input_height, input_width, batch_size, maxlen)
    if size=='full':
        model = CRNN_model(class_num,dropout,ada, input_height, input_width, batch_size, maxlen)
    if size=='smaller':
        model = smaller_model(class_num,dropout,ada, input_height, input_width, batch_size, maxlen)
    # for i in range (train_data.shape[0]):
    #     utils.showImg(tf.squeeze(train_data[i]))
    # plt.show()

    if mode=='test':
        if size=='smaller':
            model.load(smaller)
        elif size=='full':
            # name=500+i*50
            # path='history/checkpoints1/weights.%s.hdf5' % name
            model.load(best_model)
        elif size=='smaller':
            model.load(smaller)

        if cnnout:
            output = model.model.predict(test_data)
            print(output[1].shape)
            pickle.dump(output[1], open("cnnout.pkl","wb"))
            print(output[0].shape)
            ans = utils.ctc_beam_decoder(output[0])
            print(ans[0])
            print(utils.accuracy(ans[0], text))
        else:
            output = model.model.predict(test_data)
            print(output.shape)
            ans = utils.ctc_beam_decoder(output)
            print(ans[0])
            print(utils.accuracy(ans[0], text))

    if mode == 'predict':
        if size=='small':
            model.load(small)
        elif size=='full':
            model.load(best_model)
        elif size=='smaller':
            model.load(smaller)
        # model.get_rnn_weights()
        output=model.model.predict(train_data)
        model.summary()
        # model.get_weights()
        print(output.shape)
        # pickle.dump(output[1], open("cnnout.pkl","wb"))

        ans=utils.ctc_beam_decoder(output)
        print(ans[0])
        print(utils.accuracy(ans[0],text))
        utils.showImg(train_data)
        plt.show()
        # Partitions=FeatureExtraction()
        # print(Partitions.get_partitions(train_data[0]))


        # model.load(small)
        # tf.saved_model.save(model.model,'./small_model/')
        # image = np.expand_dims(image, axis=2)
        # image = np.expand_dims(image, axis=0)
        # output=model.model.predict(image)
        # print(output.shape)
        # ans = utils.ctc_beam_decoder(output)
        # print(ans[0])


    if mode =='train':
        if not new_train:
            model.load(smaller)
        model.summary()
        history = model.train(train_data,train_label,val,val_split,epochs,new_train)


    # if mode =='test':
    #     model.load('history/checkpoints/6817.hdf5')
    #     model.summary()
    #     model.test(test_data,test_label)


    if mode =='preturb':
        preturb=preturb(input_width,input_height)
        image=train_data[0]
        preturb.word_seg(image)

    # preturb.random(image,model,best_model,0.1,utils)
    # summarize history for accuracy
    # plt.subplot(2, 1, 1)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # # summarize history for loss
    # plt.subplot(2, 1, 2)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # model.save('models/my_model_'+embedding+'.h5')
    #
    # json.dump(history.history, open('history/trainHistory.json', 'w'))
    #
    # plt.show()