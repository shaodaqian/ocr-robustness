from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math

import tensorflow as tf
import tensorflow.keras.backend as k

CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'.!?,\"&£$€:\\%/@()*+"


def sparse_tuple_from(dict, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """

    indices = []
    values = []

    for n, seq in dict.items():
        indices.extend(zip([n] * seq.shape[0], [i for i in range(seq.shape[0])]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray(
        [len(dict), np.asarray(indices).max(0)[1] + 1], dtype=np.int64
    )

    return tf.SparseTensor(indices, values, shape)

def showImg(images):
    for i in range(images.shape[0]):
        image=np.squeeze(images[i])
        print(image.shape)
        plt.figure()
        plt.imshow(image, cmap='Greys')



class dist:
    def levenshtein(self,s1, s2):
        if s1[-1]==0:
            s1=s1.pop()
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


    def l2Distance(self,image1, image2):
        return math.sqrt(np.sum(np.square(np.subtract(image1, image2))))


    def l1Distance(self,image1, image2):
        return np.sum(np.absolute(np.subtract(image1, image2)))


    def l0Distance(self,image1, image2):
        return np.count_nonzero(np.absolute(np.subtract(image1, image2)))


class utils:

    def __init__(self, batch_size, data_size, CHAR_VECTOR, height, width):
        self.batch_size=batch_size
        self.data_size=data_size
        self.CHAR_VECTOR=CHAR_VECTOR
        self.height=height
        self.width=width

    def Rescale(self, image, output_size):
        (new_h,new_w) = output_size
        img= resize(image, (new_h,new_w), )
        return img


    def dense_tuple_from(self,dict, dtype=np.float32,):
        maxlen=0
        for n,seq in dict.items():
            if len(seq)>maxlen:
                maxlen=len(seq)
        maxlen+=1
        labels=np.empty(shape=[0,(maxlen+1)])
        for n, seq in dict.items():
            padding=np.zeros(maxlen)
            padding[0:len(seq)]=seq
            padding=np.append(padding,len(seq))
            padding=np.expand_dims(padding,axis=0)
            labels=np.append(labels,padding,axis=0)
        labels=labels.astype(int)
        return labels, maxlen

    def label2array(self, label, char_vector,idx):
        try:
            return [char_vector.index(x)+1 for x in label]
        except Exception as ex:
            print(label,idx)
            raise ex

    def image2array(self, label_file, root_dir):
        gt = pd.read_csv(label_file, header=None)
        gt.columns = ['img', 'label']
        size=len(gt) #3492
        size=size-4
        train_data= np.empty(shape=[0,self.height,self.width,1])
        train_labels={}
        text=[]
        val_data= np.empty(shape=[0,self.height,self.width,1])
        val_labels={}
        split=int(size*0.853)+1  # 2976 train, 512 val
        if self.data_size is not None:
            size=self.data_size
        for i in range(size):
            img_name= os.path.join(root_dir, gt.loc[i,'img'])
            image = io.imread(img_name, as_gray=True)
            image = self.Rescale(image,(self.height, self.width))
            image = np.expand_dims(image,axis=2)
            image = np.expand_dims(image,axis=0)
            train_data = np.append(train_data, image, axis=0)
            string = gt.loc[i,'label'].replace(' ','')
            text.append(string)
            item=list(string)
            item = self.label2array(item,CHAR_VECTOR,i)
            train_labels[i]=item
            # else:
            #     img_name= os.path.join(root_dir, gt.loc[i,'img'])
            #     image = io.imread(img_name, as_gray=True)
            #     image = self.Rescale(image,(self.height, self.width))
            #     image = np.expand_dims(image, axis=2)
            #     image = np.expand_dims(image,axis=0)
            #     val_data = np.append(val_data, image, axis=0)
            #     item = list(gt.loc[i,'label'].replace(' ',''))
            #     item = self.label2array(item,CHAR_VECTOR,i)
            #     val_labels[i-int(size*0.8)]=item
        train_dense, train_maxlen=self.dense_tuple_from(train_labels)
        return train_data, train_dense, train_maxlen,text

    def ground_truth_to_word(self, ground_truth):
        """
            Return the word string based on the input ground_truth
        """

        try:
            return "".join([self.CHAR_VECTOR[i-1] for i in ground_truth if i != 0])
        except Exception as ex:
            print(ground_truth)
            print(ex)
            input()

    def showImg(self,images):
        for i in range(images.shape[0]):
            image=np.squeeze(images[i])
            plt.figure()
            plt.imshow(image, cmap='gray')

    def ctc_beam_decoder(self,logits):
        length=logits.shape[0]
        pred=tf.transpose(logits,(1,0,2))
        # pred (31*batch*84)
        seq_len=[30]*length

        # test=logits[0]
        # print(test[0:3])
        # a=[]
        # for i in range (test.shape[0]):
        #     a.append(np.argmax(test[i]))
        # print(a)

        predict, logprob = tf.nn.ctc_beam_search_decoder(
                pred, seq_len, beam_width=1)
        dense_decoded = tf.sparse.to_dense(
                predict[0], name="dense_decoded"
            )
        ans=[]
        for i in range(dense_decoded.shape[0]):
            ans.append(self.ground_truth_to_word(dense_decoded[i]))
        return ans,logprob

    def accuracy(self,a,b):
        count=0
        length=len(a)
        for index,v in enumerate(a):
            if b[index]==v:
                count+=1
        return count/length


class ctc:
    def __init__(self,batch_size, max_len):
        self.max_len=max_len
        self.batch_size=batch_size
        self.frames=31
        self.cutoff=0

    def ctc_loss(self, labels, logits):
        print(labels.shape,'loss')
        if labels.shape[1] == None:
            labels = k.placeholder(shape=(self.batch_size, self.max_len+1), dtype=tf.int32)
        # tf.dtypes.cast(labels, tf.int32)
        y_true, length = tf.split(labels, [(labels.shape[1] - 1), 1], 1)
        logit_length = tf.expand_dims(tf.convert_to_tensor([self.frames-self.cutoff]*self.batch_size, dtype=tf.int32),axis=1)
        # logits=logits[:,self.cutoff:,:]
        # length = tf.squeeze(length,axis=1)
        print(y_true.shape, logits.shape, length.shape, logit_length.shape, 'ctcloss')
        return k.ctc_batch_cost(y_true, logits, logit_length, length)

    def ctc_beam_decoder_loss(self, labels, logits):
        logits=tf.transpose(logits, (1, 0, 2))
        print(labels.shape,'decode')
        if labels.shape[1] == None:
            print('init')
            return k.placeholder(shape=(1),dtype=tf.float32)
            # labels = k.placeholder(shape=(self.batch_size, self.max_len+1), dtype=tf.int32)
        y_true, length = tf.split(labels, [(labels.shape[1] - 1), 1], 1)
        print(y_true.shape, length.shape)
        length = tf.squeeze(length,axis=1)
        predict, logprob = tf.nn.ctc_beam_search_decoder(
            logits, tf.fill([self.batch_size], self.frames-self.cutoff), beam_width=100)
        dict={}
        for i in range (y_true.shape[0]):
            print(length.shape,'length')
            dict[i], dontcare =y_true[i][0:length[i]]
            print(dict[i].shape,'dict')
        sparse_true = sparse_tuple_from(dict)

        # dense_decoded = tf.sparse.to_dense(
        #         predict[0], name="dense_decoded"
        #     )
        # print(y_true.shape, dense_decoded)
        # return levenshtein(tf.cast(predict[0], tf.int32), y_true)
        return tf.reduce_mean(tf.edit_distance(tf.cast(predict[0], tf.int32), sparse_true))


# labels=tf.convert_to_tensor([[1]])
# logits=tf.convert_to_tensor([[[0.3,0.3,0.3]],[[0.3,0.3,0.3]],[[0.9,0.1,0.0]],[[0.9,0.1,0.0]]])
# print(tf.nn.ctc_loss(labels,logits,tf.convert_to_tensor([1]),[20]))


# print(utils.levenshtein([1,2,4,4,2,6,7],[9,2,4,4,8,6]))
# a=np.array([[1,2,3],[4,5,6]])
# b,c = np.split(a,[-1],axis=1)
# print(b)