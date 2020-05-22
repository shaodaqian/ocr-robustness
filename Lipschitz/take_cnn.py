import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
import matplotlib as plt

class smaller_model(object):
    def __init__(self, num_of_classes, dropout_p,
                 optimizer, height, width, batch_size):
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.height = height
        self.width = width
        self.batch_size=batch_size
        self._build_model()

    def _build_model(self):
        input=keras.Input(shape=(self.height, self.width, 1))
        conv1=layers.Conv2D(64, (7,7), strides=(2, 2), padding='same', activation='relu')(input) #(batch,16,64,64)
        pool1=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv1)
        conv2=layers.Conv2D(64, (3,3), padding='same', activation='relu')(pool1) #(batch,8,32,64)
        conv3=layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv2) #(batch,8,32,64)
        conv4=layers.Conv2D(128, (3,3), strides=(2,1), padding='same', activation='relu')(conv3) #(batch,4,32,128)
        conv5=layers.Conv2D(128, (3,3), padding='same', activation='relu')(conv4) #(batch,4,32,128)
        conv6=layers.Conv2D(256, (3,3), strides=(2,1), padding='same', activation='relu')(conv5) #(batch,2,32,256)
        bn1=layers.BatchNormalization()(conv6)
        conv7=layers.Conv2D(256, (3,3), padding='same', activation='relu')(bn1) #(batch,2,32,256)
        bn2=layers.BatchNormalization()(conv7)
        CNN_out=layers.Conv2D(256, (2,2), padding='valid', activation='relu')(bn2) #(batch,1,31,256)

        reshape=layers.Reshape((31,256))(CNN_out) #(batch,31,512)

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
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)#,metrics=[self.loss.ctc_beam_decoder_loss])
        # target_tensors = self.train_label

class cnn_model(object):
    def __init__(self, num_of_classes, dropout_p,
                 optimizer, height, width,batch_size):
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.height = height
        self.width = width
        self.batch_size=batch_size
        self._build_model()

    def _build_model(self):
        input=keras.Input(shape=(self.height, self.width, 1))
        conv1=layers.Conv2D(64, (7,7), strides=(2, 2), padding='same', activation='relu')(input) #(batch,16,64,64)
        pool1=layers.MaxPool2D(pool_size=(2,2),strides=2, padding='same')(conv1)
        conv2=layers.Conv2D(64, (3,3), padding='same', activation='relu')(pool1) #(batch,8,32,64)
        conv3=layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv2) #(batch,8,32,64)
        conv4=layers.Conv2D(128, (3,3), strides=(2,1), padding='same', activation='relu')(conv3) #(batch,4,32,128)
        conv5=layers.Conv2D(128, (3,3), padding='same', activation='relu')(conv4) #(batch,4,32,128)
        conv6=layers.Conv2D(256, (3,3), strides=(2,1), padding='same', activation='relu')(conv5) #(batch,2,32,256)
        bn1=layers.BatchNormalization()(conv6)
        conv7=layers.Conv2D(256, (3,3), padding='same', activation='relu')(bn1) #(batch,2,32,256)
        bn2=layers.BatchNormalization()(conv7)
        CNN_out=layers.Conv2D(256, (2,2), padding='valid', activation='relu')(bn2) #(batch,1,31,256)
        reshape=layers.Reshape((31,256))(CNN_out) #(batch,31,512)

        # decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        #     logits, [6,5])
        # dense_decoded = tf.sparse.to_dense(
        #     decoded[0], default_value=-1, name="dense_decoded"
        # )


        self.model=keras.Model(inputs=input, outputs=reshape)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)#,metrics=[self.loss.ctc_beam_decoder_loss])
        # target_tensors = self.train_label

if __name__ == '__main__':
    class_num=84
    dropout=0.2
    ada=tf.keras.optimizers.Adam(learning_rate=0.0001)
    input_height, input_width, batch_size=32,128,32
    full_model = smaller_model(class_num, dropout, ada, input_height, input_width, batch_size)
    cnn=cnn_model(class_num, dropout, ada, input_height, input_width, batch_size)
    full_model.model.load_weights('model/256rnn71.hdf5')
    # full_model=models.load_model('model/256rnn71.hdf5')
    temp_weights = [layer.get_weights() for layer in full_model.model.layers]
    print(len(temp_weights))
    i = 0
    for layer in cnn.model.layers:
        print(i)
        print(layer.name)
        layer.set_weights(temp_weights[i])
        i=i+1

    models.save_model(cnn.model,'model/cnn.h5py',save_format='h5')
    # keras_model = models.load_model('model/cnn.h5py')

    # cnn.model.save('model/cnn.hdf5')