import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
from utils import ctc
import matplotlib as plt

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer

# Other
import re
import string
import numpy as np
import pandas as pd
import shutil
import os
import pickle
# from sklearn.manifold import TSNE

class CRNN_model(object):
    def __init__(self, num_of_classes, dropout_p,
                 optimizer, height, width, batch_size, train_maxlen):
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.height = height
        self.width = width
        self.batch_size=batch_size
        self.loss=ctc(batch_size,train_maxlen)
        self._build_model()

    def _build_model(self):
        input=keras.Input(shape=(self.height, self.width, 1))
        conv1=layers.Conv2D(64, (3,3), padding='same', activation='relu')(input) #(batch,32,128,64)
        pool1=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv1)
        conv2=layers.Conv2D(128, (3,3), padding='same', activation='relu')(pool1) #(batch,16,64,128)
        pool2=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv2)
        conv3=layers.Conv2D(256, (3,3), padding='same', activation='relu')(pool2) #(batch,8,32,256)
        bn1=layers.BatchNormalization()(conv3)
        dp1=layers.Dropout(0.4)(bn1)
        conv4=layers.Conv2D(256, (3,3), padding='same', activation='relu')(dp1)
        pool3=layers.MaxPool2D(pool_size=(2,2),strides=(2,1), padding='same')(conv4)
        dp4=layers.Dropout(0.4)(pool3)
        conv5=layers.Conv2D(512, (3,3), padding='same', activation='relu')(dp4) #(batch,4,32,256)
        bn2=layers.BatchNormalization()(conv5)
        # spec
        dp2=layers.Dropout(0.4)(bn2)
        conv6=layers.Conv2D(512, (3,3), padding='same', activation='relu')(dp2)
        bn3=layers.BatchNormalization()(conv6)
        dp3=layers.Dropout(0.4)(bn3)
        pool4=layers.MaxPool2D(pool_size=(2,2),strides=(2,1), padding='same')(dp3) #(batch,2,32,512)
        CNN_out=layers.Conv2D(512, (2,2), padding='valid', activation='relu')(pool4) #(batch,1,31,512)

        reshape=layers.Reshape((31,512))(CNN_out) #(batch,31,512)
        # print(RNN_in.shape)
        RNN_in=layers.Dropout(0.5)(reshape)
        lstm1=layers.Bidirectional(layers.LSTM(256, return_sequences=True), backward_layer=layers.LSTM(256,return_sequences=True, go_backwards=True))(RNN_in) #(batch,31,512)
        # lstm1=layers.LSTM(256, return_sequences=True, activation='sigmoid')(RNN_in)
        drop1=layers.Dropout(0.5)(lstm1)
        lstm2=layers.Bidirectional(layers.LSTM(256, return_sequences=True), backward_layer=layers.LSTM(256,return_sequences=True, go_backwards=True))(drop1)
        # lstm2=layers.LSTM(256, return_sequences=True, activation='sigmoid')(bn4)
        drop2=layers.Dropout(0.4)(lstm2)
        logits=layers.Dense(self.num_of_classes, activation='softmax'
                            ,kernel_constraint='UnitNorm', use_bias=False)(drop2) #(batch,24,84)
        # decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        #     logits, [6,5])
        # dense_decoded = tf.sparse.to_dense(
        #     decoded[0], default_value=-1, name="dense_decoded"
        # )

        self.model=keras.Model(inputs=input, outputs=logits)
        self.model.compile(loss=self.loss.ctc_loss, optimizer=self.optimizer)#,metrics=[self.loss.ctc_beam_decoder_loss])

    def train(self,training_inputs, training_labels, val, val_split,
              epochs, new):
        if new:
            if os.path.exists('tensor_board') and os.path.isdir('tensor_board'):
                shutil.rmtree('tensor_board',ignore_errors=True)
                print('new tensorboard')
        else:
            print('existing tensorboard')
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='tensor_board', histogram_freq=1, write_images=True)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            'history/checkpoints1/weights.{epoch:02d}.hdf5', period=50)
        early_stopping_checkpoint = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

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

    def save(self,path):
        self.model.save(path)
        print("neural network saved to" + path)

    def load(self,path):
        self.model.load_weights(path)
        print("neural network loaded from" + path)

    def predict(self,x):
        x = np.expand_dims(x, axis=0)
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

    def get_weights(self):
        for layer in self.model.layers:
            weights=np.array(layer.get_weights())
            print(layer.trainable_weights)
            # print(weights.shape)
            # print(weights)



class small_model(object):
    def __init__(self, num_of_classes, dropout_p,
                 optimizer, height, width,batch_size, train_maxlen):
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.height = height
        self.width = width
        self.batch_size=batch_size
        self.loss=ctc(batch_size,train_maxlen)
        self._build_model()

    def _build_model(self):
        input=keras.Input(shape=(self.height, self.width, 1))
        conv1=layers.Conv2D(64, (7,7), strides=(2, 2), padding='same', activation='relu')(input) #(batch,16,64,64)
        pool1=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv1)
        conv2=layers.Conv2D(64, (3,3), padding='same', activation='relu')(pool1) #(batch,8,32,64)
        dp=layers.Dropout(0.4)(conv2)
        conv3=layers.Conv2D(64, (3,3), padding='same', activation='relu')(dp) #(batch,8,32,64)
        conv4=layers.Conv2D(128, (3,3), strides=(2,1), padding='same', activation='relu')(conv3) #(batch,4,32,128)
        dp1=layers.Dropout(0.4)(conv4)
        conv5=layers.Conv2D(128, (3,3), padding='same', activation='relu')(dp1) #(batch,4,32,128)
        conv6=layers.Conv2D(256, (3,3), strides=(2,1), padding='same', activation='relu')(conv5) #(batch,2,32,256)
        bn1=layers.BatchNormalization()(conv6)
        dp2=layers.Dropout(0.4)(bn1)
        conv7=layers.Conv2D(256, (3,3), padding='same', activation='relu')(dp2) #(batch,2,32,256)
        bn2=layers.BatchNormalization()(conv7)
        dp4=layers.Dropout(0.4)(bn2)
        CNN_out=layers.Conv2D(256, (2,2), padding='valid', activation='relu')(dp4) #(batch,1,31,256)

        reshape=layers.Reshape((31,256))(CNN_out) #(batch,31,256)

        RNN_in=layers.Dropout(0.4)(reshape)
        # lstm1=layers.Bidirectional(layers.LSTM(31, return_sequences=True), backward_layer=layers.LSTM(31,return_sequences=True, go_backwards=True))(RNN_in) #(batch,31,512)
        rnn=layers.SimpleRNN(256, return_sequences=True)(RNN_in)
        # drop1=layers.Dropout(0.5)(lstm1)
        # lstm2=layers.Bidirectional(layers.LSTM(31, return_sequences=True), backward_layer=layers.LSTM(31,return_sequences=True, go_backwards=True))(Drop1)
        # lstm1=layers.LSTM(256, return_sequences=True, activation='sigmoid')(RNN_in)
        # lstm2=layers.LSTM(256, return_sequences=True, activation='sigmoid')(bn4)
        drop2=layers.Dropout(0.4)(rnn)
        logits=layers.Dense(self.num_of_classes, activation='softmax'
                            ,kernel_constraint='UnitNorm', use_bias=False
                            )(drop2) #(batch,31,84)

        # decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        #     logits, [6,5])
        # dense_decoded = tf.sparse.to_dense(
        #     decoded[0], default_value=-1, name="dense_decoded"
        # )


        self.model=keras.Model(inputs=input, outputs=logits)
        self.model.compile(loss=self.loss.ctc_loss, optimizer=self.optimizer)#,metrics=[self.loss.ctc_beam_decoder_loss])
        # target_tensors = self.train_label

    def train(self,training_inputs, training_labels, val, val_split,
              epochs, new):
        if new:
            if os.path.exists('tensor_board') and os.path.isdir('tensor_board'):
                shutil.rmtree('tensor_board',ignore_errors=True)
                print('new tensorboard')
        else:
            print('existing tensorboard')
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir='tensor_board', histogram_freq=1, write_images=True)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            'history/smaller_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=20)
        early_stopping_checkpoint = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

        history = self.model.fit(training_inputs, training_labels,
                        # validation_data=val,
                        validation_split=val_split,
                        epochs=epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        verbose=2,
                        callbacks=[model_checkpoint_callback])
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

    def save(self,path):
        self.model.save(path)
        print("neural network saved to" + path)

    def load(self,path):
        self.model.load_weights(path)
        print("neural network loaded from" + path)

    def predict(self,x):
        x = np.expand_dims(x, axis=0)
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

    def get_rnn_weights(self):
        f = open("rnn.pkl","wb")
        a=[]
        for layer in self.model.layers:
            print(layer.name)
            if len(layer.get_weights())>0:
                print(np.array(layer.get_weights()[0]).shape)
            if layer.name=='simple_rnn':
                for i in layer.trainable_weights:
                    # a=tf.convert_to_tensor(i,dtype=tf.float32)
                    # print(i)
                    a.append(i.numpy())
            if layer.name=='dense':
                for i in layer.trainable_weights:
                    # print(i)
                    a.append(i.numpy())
        pickle.dump(a, f)
        f.close()



class smaller_model(object):
    def __init__(self, num_of_classes, dropout_p,
                 optimizer, height, width,batch_size, train_maxlen):
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.height = height
        self.width = width
        self.batch_size=batch_size
        self.loss=ctc(batch_size,train_maxlen)
        self._build_model()

    def _build_model(self):
        input=keras.Input(shape=(self.height, self.width, 1))
        conv1=layers.Conv2D(64, (3,3), padding='same', activation='relu')(input) #(batch,32,128,64)
        pool1=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv1)
        conv2=layers.Conv2D(128, (3,3), padding='same', activation='relu')(pool1) #(batch,16,64,128)
        pool2=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv2)#(batch,8,32,128)
        trans=layers.Permute((2,1,3))(pool2)#(batch,32,8,128)
        reshape=layers.Reshape((32,1024))(trans) #(batch,32,1024)

        RNN_in=layers.Dropout(0.5)(reshape)
        # lstm1=layers.Bidirectional(layers.LSTM(31, return_sequences=True), backward_layer=layers.LSTM(31,return_sequences=True, go_backwards=True))(RNN_in) #(batch,31,512)
        rnn=layers.SimpleRNN(256, return_sequences=True)(RNN_in)
        # drop1=layers.Dropout(0.5)(lstm1)
        # lstm2=layers.Bidirectional(layers.LSTM(31, return_sequences=True), backward_layer=layers.LSTM(31,return_sequences=True, go_backwards=True))(Drop1)
        # lstm1=layers.LSTM(256, return_sequences=True, activation='sigmoid')(RNN_in)
        # lstm2=layers.LSTM(256, return_sequences=True, activation='sigmoid')(bn4)
        drop2=layers.Dropout(0.5)(rnn)
        logits=layers.Dense(self.num_of_classes, activation='softmax'
                            ,kernel_constraint='UnitNorm', use_bias=False
                            )(drop2) #(batch,31,84)

        # decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        #     logits, [6,5])
        # dense_decoded = tf.sparse.to_dense(
        #     decoded[0], default_value=-1, name="dense_decoded"
        # )


        self.model=keras.Model(inputs=input, outputs=logits)
        self.model.compile(loss=self.loss.ctc_loss, optimizer=self.optimizer)#,metrics=[self.loss.ctc_beam_decoder_loss])
        # target_tensors = self.train_label

    def train(self,training_inputs, training_labels, val, val_split,
              epochs, new):
        if new:
            if os.path.exists('tensor_board') and os.path.isdir('tensor_board'):
                shutil.rmtree('tensor_board',ignore_errors=True)
                print('new tensorboard')
        else:
            print('existing tensorboard')
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir='tensor_board', histogram_freq=1, write_images=True)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            'history/smaller_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=20)
        early_stopping_checkpoint = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

        history = self.model.fit(training_inputs, training_labels,
                        # validation_data=val,
                        validation_split=val_split,
                        epochs=epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        verbose=2,
                        callbacks=[model_checkpoint_callback])
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

    def save(self,path):
        self.model.save(path)
        print("neural network saved to" + path)

    def load(self,path):
        self.model.load_weights(path)
        print("neural network loaded from" + path)

    def predict(self,x):
        x = np.expand_dims(x, axis=0)
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

    def get_rnn_weights(self):
        f = open("rnn.pkl","wb")
        a=[]
        for layer in self.model.layers:
            print(layer.name)
            if len(layer.get_weights())>0:
                print(np.array(layer.get_weights()[0]).shape)
            if layer.name=='simple_rnn':
                for i in layer.trainable_weights:
                    # a=tf.convert_to_tensor(i,dtype=tf.float32)
                    # print(i)
                    a.append(i.numpy())
            if layer.name=='dense':
                for i in layer.trainable_weights:
                    # print(i)
                    a.append(i.numpy())
        pickle.dump(a, f)
        f.close()

                # print(weights.shape)
            # print(weights)