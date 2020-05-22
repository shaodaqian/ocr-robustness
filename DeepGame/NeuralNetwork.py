# -*- coding: utf-8 -*-
"""
Construct a NeuralNetwork class to include operations
related to various datasets and corresponding models.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

import copy

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import shutil
# from keras.preprocessing import image as Image
import cv2


from basics import assure_path_exists
from DataSet import *
from utils import utils
from utils import ctc


CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'.!?,\"&£$€:\\%/@()*+"


class NeuralNetwork():
    def __init__(self, data_set):
        self.data_set = data_set
        self.num_of_classes = 84
        self.optimizer = 'Adadelta'
        self.regParam = 0.001
        self.height = 32
        self.width = 128
        self.batch_size=32
        self.loss=ctc(32,10)
        self.util=utils(32,1,CHAR_VECTOR,32,128)
        assure_path_exists("%s_pic/" % self.data_set)
        self._build_model()

    def _build_model(self):
        if self.data_set=='crnn':
            input=tf.keras.Input(shape=(self.height, self.width, 1))
            conv1=layers.Conv2D(64, (3,3), padding='same', activation='relu')(input) #(batch,32,128,64)
            pool1=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv1)
            conv2=layers.Conv2D(128, (3,3), padding='same', activation='relu')(pool1) #(batch,16,64,128)
            pool2=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv2)
            conv3=layers.Conv2D(256, (3,3), padding='same', activation='relu')(pool2) #(batch,8,32,256)
            bn1=layers.BatchNormalization()(conv3)
            conv4=layers.Conv2D(256, (3,3), padding='same', activation='relu')(bn1)
            pool3=layers.MaxPool2D(pool_size=(2,2),strides=(2,1), padding='same')(conv4)
            conv5=layers.Conv2D(512, (3,3), padding='same', activation='relu')(pool3) #(batch,4,32,256)
            bn2=layers.BatchNormalization()(conv5)
            conv6=layers.Conv2D(512, (3,3), padding='same', activation='relu')(bn2)
            bn3=layers.BatchNormalization()(conv6)
            pool4=layers.MaxPool2D(pool_size=(2,2),strides=(2,1), padding='same')(bn3) #(batch,2,32,512)
            CNN_out=layers.Conv2D(512, (2,2), padding='valid', activation='relu')(pool4) #(batch,1,31,512)

            RNN_in=layers.Reshape((31,512))(CNN_out) #(batch,31,512)
            # print(RNN_in.shape)

            lstm1=layers.Bidirectional(layers.LSTM(256, return_sequences=True), backward_layer=layers.LSTM(256,return_sequences=True, go_backwards=True))(RNN_in) #(batch,31,512)
            # lstm1=layers.LSTM(256, return_sequences=True, activation='sigmoid')(RNN_in)
            lstm2=layers.Bidirectional(layers.LSTM(256, return_sequences=True), backward_layer=layers.LSTM(256,return_sequences=True, go_backwards=True))(lstm1)
            # lstm2=layers.LSTM(256, return_sequences=True, activation='sigmoid')(bn4)
            logits=layers.Dense(self.num_of_classes, activation='softmax'
                                ,kernel_constraint='UnitNorm', use_bias=False
                                )(lstm2) #(batch,24,84)

            self.model=tf.keras.Model(inputs=input, outputs=logits)
            self.model.compile(loss=self.loss.ctc_loss, optimizer=self.optimizer)#,metrics=[self.loss.ctc_beam_decoder_loss])

        if self.data_set=='small':
            input = tf.keras.Input(shape=(self.height, self.width, 1))
            conv1 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(
                input)  # (batch,16,64,64)
            pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
            conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)  # (batch,8,32,64)
            dp = layers.Dropout(0.4)(conv2)
            conv3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(dp)  # (batch,8,32,64)
            conv4 = layers.Conv2D(128, (3, 3), strides=(2, 1), padding='same', activation='relu')(
                conv3)  # (batch,4,32,128)
            dp1 = layers.Dropout(0.4)(conv4)
            conv5 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(dp1)  # (batch,4,32,128)
            conv6 = layers.Conv2D(256, (3, 3), strides=(2, 1), padding='same', activation='relu')(
                conv5)  # (batch,2,32,256)
            bn1 = layers.BatchNormalization()(conv6)
            dp2 = layers.Dropout(0.4)(bn1)
            conv7 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(dp2)  # (batch,2,32,256)
            bn2 = layers.BatchNormalization()(conv7)
            dp4 = layers.Dropout(0.4)(bn2)
            CNN_out = layers.Conv2D(256, (2, 2), padding='valid', activation='relu')(dp4)  # (batch,1,31,256)

            reshape = layers.Reshape((31, 256))(CNN_out)  # (batch,31,256)

            RNN_in = layers.Dropout(0.4)(reshape)
            # lstm1=layers.Bidirectional(layers.LSTM(31, return_sequences=True), backward_layer=layers.LSTM(31,return_sequences=True, go_backwards=True))(RNN_in) #(batch,31,512)
            rnn = layers.SimpleRNN(256, return_sequences=True)(RNN_in)
            drop2 = layers.Dropout(0.4)(rnn)
            logits = layers.Dense(self.num_of_classes, activation='softmax'
                                  , kernel_constraint='UnitNorm', use_bias=False
                                  )(drop2)  # (batch,31,84)

            self.model = tf.keras.Model(inputs=input, outputs=logits)
            self.model.compile(loss=self.loss.ctc_loss,
                               optimizer=self.optimizer)  # ,metrics=[self.loss.ctc_beam_decoder_loss])
            # input = tf.keras.Input(shape=(self.height, self.width, 1))
            # conv1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)  # (batch,32,128,64)
            # pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
            # conv2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)  # (batch,16,64,128)
            # pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(conv2)  # (batch,8,32,128)
            # trans = layers.Permute((2, 1, 3))(pool2)  # (batch,32,8,128)
            # reshape = layers.Reshape((32, 1024))(trans)  # (batch,32,1024)
            #
            # RNN_in = layers.Dropout(0.5)(reshape)
            # # lstm1=layers.Bidirectional(layers.LSTM(31, return_sequences=True), backward_layer=layers.LSTM(31,return_sequences=True, go_backwards=True))(RNN_in) #(batch,31,512)
            # rnn = layers.SimpleRNN(256, return_sequences=True)(RNN_in)
            # # drop1=layers.Dropout(0.5)(lstm1)
            # # lstm2=layers.Bidirectional(layers.LSTM(31, return_sequences=True), backward_layer=layers.LSTM(31,return_sequences=True, go_backwards=True))(Drop1)
            # # lstm1=layers.LSTM(256, return_sequences=True, activation='sigmoid')(RNN_in)
            # # lstm2=layers.LSTM(256, return_sequences=True, activation='sigmoid')(bn4)
            # drop2 = layers.Dropout(0.5)(rnn)
            # logits = layers.Dense(self.num_of_classes, activation='softmax'
            #                       , kernel_constraint='UnitNorm', use_bias=False
            #                       )(drop2)  # (batch,31,84)
            #
            # # decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            # #     logits, [6,5])
            # # dense_decoded = tf.sparse.to_dense(
            # #     decoded[0], default_value=-1, name="dense_decoded"
            # # )
            #
            # self.model = tf.keras.Model(inputs=input, outputs=logits)
            # self.model.compile(loss=self.loss.ctc_loss,
            #                    optimizer=self.optimizer)  # ,metrics=[self.loss.ctc_beam_decoder_loss])
            # # target_tensors = self.train_label



    def train(self,training_inputs, training_labels, val, val_split,
              epochs, new):
        if new:
            if os.path.exists('tensor_board') and os.path.isdir('tensor_board'):
                shutil.rmtree('tensor_board',ignore_errors=True)
                print('new tensorboard')
        else:
            print('existing tensorboard')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='tensor_board', histogram_freq=1, write_images=True)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            'history/training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=10)
        early_stopping_checkpoint = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

        history = self.model.fit(training_inputs, training_labels,
                        # validation_data=val,
                        validation_split=val_split,
                        epochs=epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        verbose=2,
                        callbacks=[tensorboard_callback,
                                   model_checkpoint_callback])
        return history

    def test(self, testing_inputs, testing_labels):
        """
        Testing function

        Args:
            testing_inputs (numpy.ndarray): Testing set inputs
            testing_labels (numpy.ndarray): Testing set labels
            batch_size (int): Batch size

        Returns: None

        """
        # Evaluate inputs
        self.model.evaluate(testing_inputs, testing_labels, batch_size=self.batch_size, verbose=2)
        # self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)

    def save_network(self,path):
        self.model.save(path)
        print("neural network saved to" + path)

    def load_network(self):
        if self.data_set=='crnn':
            self.model.load_weights('models/crnn.hdf5')
        if self.data_set=='small':
            self.model.load_weights('models/crnnsmall.hdf5')
        print("neural network loaded")

    def predict(self,x):
        image = np.expand_dims(x, axis=0)
        predict_value=self.model.predict(image)
        ans,logprob=self.util.ctc_beam_decoder(predict_value)
        return ans[0],logprob

    def predict_all(self,x):
        predict_value=self.model.predict(x)
        ans,logprob=self.util.ctc_beam_decoder(predict_value)
        return ans,logprob

    def save_input(self, image, filename, mul=1):
        # image = Image.array_to_img(image.copy())
        image1 = copy.deepcopy(image)
        m=np.amin(image1)
        x=np.squeeze(image1)
        x=(x-m)/(1-m)
        cv2.imwrite(filename, x *255.0*mul ,[cv2.IMWRITE_PNG_COMPRESSION,0])

        # plt.imsave(filename, x)
        # causes discrepancy
        # image_cv = copy.deepcopy(image)
        # cv2.imwrite(filename, image_cv * 255.0, [cv2.IMWRITE_PNG_COMPRESSION, 9])


    # # Get softmax logits, i.e., the inputs to the softmax function of the classification layer,
    # # as softmax probabilities may be too close to each other after just one pixel manipulation.
    # def softmax_logits(self, manipulated_images, batch_size=512):
    #     model = self.model
    #
    #     func = K.function([model.layers[0].input] + [K.learning_phase()],
    #                       [model.layers[model.layers.__len__() - 1].output.op.inputs[0]])
    #
    #     # func = K.function([model.layers[0].input] + [K.learning_phase()],
    #     #                   [model.layers[model.layers.__len__() - 1].output])
    #
    #     if len(manipulated_images) >= batch_size:
    #         softmax_logits = []
    #         batch, remainder = divmod(len(manipulated_images), batch_size)
    #         for b in range(batch):
    #             logits = func([manipulated_images[b * batch_size:(b + 1) * batch_size], 0])[0]
    #             softmax_logits.append(logits)
    #         softmax_logits = np.asarray(softmax_logits)
    #         softmax_logits = softmax_logits.reshape(batch * batch_size, model.output_shape[1])
    #         # note that here if logits is empty, it is fine, as it won't be concatenated.
    #         logits = func([manipulated_images[batch * batch_size:len(manipulated_images)], 0])[0]
    #         softmax_logits = np.concatenate((softmax_logits, logits), axis=0)
    #     else:
    #         softmax_logits = func([manipulated_images, 0])[0]
    #
    #     # softmax_logits = func([manipulated_images, 0])[0]
    #     # print(softmax_logits.shape)
    #     return softmax_logits
