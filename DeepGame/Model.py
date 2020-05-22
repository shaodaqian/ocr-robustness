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
# from sklearn.manifold import TSNE

class CRNN_model(object):
    def __init__(self, num_of_classes, dropout_p,
                 optimizer, regParam, height, width, train_label,batch_size, train_maxlen):
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.regParam = regParam
        self.height = height
        self.width = width
        self.train_label=train_label
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
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='tensor_board', histogram_freq=1, write_images=True)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            'history/training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=10)
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
        return self.model.predict(x, batch_size=1)

    def summary(self):
        self.model.summary()

    def get_weights(self):
        for layer in self.model.layers:
            weights=np.array(layer.get_weights())
            print(layer.trainable_weights)
            # print(weights.shape)
            # print(weights)