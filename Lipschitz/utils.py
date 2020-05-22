#Common utility file
#From Fast-Lin repository
#https://github.com/huanzhang12/CertifiedReLURobustness

import numpy as np
import random
import os
import pandas as pd
from PIL import Image
from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean
import tensorflow as tf
random.seed(1215)
np.random.seed(1215)
CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'.!?,\"&£$€:\\%/@()*+"


def linf_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=np.inf)

def l2_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=2)

def l1_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=1)

def l0_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=0)

def show(img, name = "output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save('img', img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    return
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def generate_data(data, samples, targeted=True, random_and_least_likely = False, skip_wrong_label = True, start=0, ids = None, 
        target_classes = None, target_type = 0b1111, predictor = None, imagenet=False, remove_background_class=False, save_inputs=False, model_name=None, save_inputs_dir=None):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    ids: true IDs of images in the dataset, if given, will use these images
    target_classes: a list of list of labels for each ids
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    true_labels = []
    true_ids = []
    information = []
    target_candidate_pool = np.eye(data.test_labels.shape[1])
    target_candidate_pool_remove_background_class = np.eye(data.test_labels.shape[1] - 1)
    print('generating labels...')
    if ids is None:
        ids = range(samples)
    else:
        ids = ids[start:start+samples]
        if target_classes:
            target_classes = target_classes[start:start+samples]
        start = 0
    total = 0
    for i in ids:
        total += 1
        if targeted:
            predicted_label = -1 # unknown
            if random_and_least_likely:
                # if there is no user specified target classes
                if target_classes is None:
                    original_predict = np.squeeze(predictor(np.array([data.test_data[start+i]])))
                    num_classes = len(original_predict)
                    predicted_label = np.argmax(original_predict)
                    least_likely_label = np.argmin(original_predict)
                    top2_label = np.argsort(original_predict)[-2]
                    start_class = 1 if (imagenet and not remove_background_class) else 0
                    random_class = predicted_label
                    new_seq = [least_likely_label, top2_label, predicted_label]
                    while random_class in new_seq:
                        random_class = random.randint(start_class, start_class + num_classes - 1)
                    new_seq[2] = random_class
                    true_label = np.argmax(data.test_labels[start+i])
                    seq = []
                    if true_label != predicted_label and skip_wrong_label:
                        seq = []
                    else:
                        if target_type & 0b10000:
                            for c in range(num_classes):
                                if c != predicted_label:
                                    seq.append(c)
                                    information.append('class'+str(c))
                        else:
                            if target_type & 0b0100:
                                # least
                                seq.append(new_seq[0])
                                information.append('least')
                            if target_type & 0b0001:
                                # top-2
                                seq.append(new_seq[1])
                                information.append('top2')
                            if target_type & 0b0010:
                                # random
                                seq.append(new_seq[2])
                                information.append('random')
                else:
                    # use user specified target classes
                    seq = target_classes[total - 1]
                    information.extend(len(seq) * ['user'])
            else:
                if imagenet:
                    if remove_background_class:
                        seq = random.sample(range(0,1000), 10)
                    else:
                        seq = random.sample(range(1,1001), 10)
                    information.extend(data.test_labels.shape[1] * ['random'])
                else:
                    seq = range(data.test_labels.shape[1])
                    information.extend(data.test_labels.shape[1] * ['seq'])
            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}, info = {}".format(total, start + i, 
                np.argmax(data.test_labels[start+i]), predicted_label, np.argmax(data.test_labels[start+i]) == predicted_label, seq, [] if len(seq) == 0 else information[-len(seq):]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])):
                    continue
                inputs.append(data.test_data[start+i])
                if remove_background_class:
                    targets.append(target_candidate_pool_remove_background_class[j])
                else:
                    targets.append(target_candidate_pool[j])
                true_labels.append(data.test_labels[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
        else:
            true_label = np.argmax(data.test_labels[start+i])
            original_predict = np.squeeze(predictor(np.array([data.test_data[start+i]])))
            num_classes = len(original_predict)
            predicted_label = np.argmax(original_predict) 
            if true_label != predicted_label and skip_wrong_label:
                continue
            else:
                inputs.append(data.test_data[start+i])
                if remove_background_class:
                    # shift target class by 1
                    print(np.argmax(data.test_labels[start+i]))
                    print(np.argmax(data.test_labels[start+i][1:1001]))
                    targets.append(data.test_labels[start+i][1:1001])
                else:
                    targets.append(data.test_labels[start+i])
                true_labels.append(data.test_labels[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
                information.extend(['original'])

    inputs = np.array(inputs)
    targets = np.array(targets)
    true_labels = np.array(true_labels)
    true_ids = np.array(true_ids)
    print('labels generated')
    print('{} images generated in total.'.format(len(inputs)))
    if save_inputs:
        if not os.path.exists(save_inputs_dir):
            os.makedirs(save_inputs_dir)
        save_model_dir = os.path.join(save_inputs_dir,model_name)
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        info_set = list(set(information))
        for info_type in info_set:
            save_type_dir = os.path.join(save_model_dir,info_type)
            if not os.path.exists(save_type_dir):
                os.makedirs(save_type_dir)
            counter = 0
            for i in range(len(information)):
                if information[i] == info_type:
                    df = inputs[i,:,:,0]
                    df = df.flatten()
                    np.savetxt(os.path.join(save_type_dir,'point{}.txt'.format(counter)),df,newline='\t')
                    counter += 1
            target_labels = np.array([np.argmax(targets[i]) for i in range(len(information)) if information[i]==info_type])
            np.savetxt(os.path.join(save_model_dir,model_name+'_target_'+info_type+'.txt'),target_labels,fmt='%d',delimiter='\n') 
    return inputs, targets, true_labels, true_ids, information

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
