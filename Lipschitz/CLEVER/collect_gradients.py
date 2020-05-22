#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_gradients.py

Front end for collecting maximum gradient norm samples

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""

from __future__ import division

import glob
import numpy as np
import scipy.io as sio
import random
import time
import sys
import os
from functools import partial
from multiprocessing import Pool
import scipy
from scipy.stats import weibull_min
import scipy.optimize
from utils import generate_data, utils
import csv


from estimate_gradient_norm import EstimateLipschitz
from clever import get_lipschitz_estimate, parse_filename
from utils import generate_data

def collect_gradients(dataset, model_name, norm, numimg=10):
    random.seed(1215)
    np.random.seed(1215)

    # create output directory
    os.system("mkdir -p {}/{}_{}".format('lipschitz_mat', dataset, model_name))
    
    # create a Lipschitz estimator class (initial it early to save multiprocessing memory)
    ids = None
    target_classes = None
    target_type = 0b0010

    idx = 32
    # frame=1

    Nsamp = 1024
    Niters = 250
    
    import tensorflow as tf
    # from setup_cifar import CIFAR
    # from setup_mnist import MNIST
    # from setup_tinyimagenet import tinyImagenet

    np.random.seed(1215)
    tf.random.set_seed(1215)
    random.seed(1215)
    # returns the input tensor and output prediction vector
    frame=0
    bounds = []
    CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'.!?,\"&£$€:\\%/@()*+"
    largest_lip=0
    smallest_adv=1000
    for i in range (31):
        for j in range(10):
            # for j in range(len(CHAR_VECTOR)):

            clever_estimator = EstimateLipschitz(nthreads=0)
            model = clever_estimator.load_model(dataset, batch_size = 0, compute_slope = False, order = 1)
            # load dataset
            # datasets_loader = {"mnist": MNIST, "cifar": CIFAR, "tinyimagenet": tinyImagenet}
            # data = datasets_loader[dataset]()
            # for prediction
            # generate target images

            data_size = 100
            input_height, input_width = 32, 128
            train_label_path = '../crnndata/ICDAR_C1_training/gt.csv'
            train_root_dir = '../crnndata/ICDAR_C1_training'
            test_label_path = '../crnndata/ICDAR_C1_testing/gt.csv'
            test_root_dir = '../crnndata/ICDAR_C1_testing'
            utilities = utils(32, data_size, CHAR_VECTOR, input_height, input_width)
            x, y, maxlen, text = utilities.image2array(test_label_path, test_root_dir)
            x = x[idx:idx + 1, :, :]
            input=x
            # preds = model.model.predict(inputs)
            # true_label=np.argmax(preds[0][frame])
            timestart = time.time()

            print("got {} images".format(input.shape))
            total = 0
            # original_predict = np.squeeze(sess.run(output, feed_dict = {img: [input_img]}))
            # print(input_img.shape)
            print("processing image {}".format(i))
            original_predict = model.predict(input)
            true_label = np.argmax(original_predict[0][frame])
            predicted_label = true_label
            if predicted_label==int(j):
                print("same class as predicted")
                continue
            least_likely_label = np.argmin(original_predict[0][frame])
            original_prob = np.sort(original_predict[0][frame])
            original_class = np.argsort(original_predict[0][frame])
            # print("Top-10 classifications:", original_class[-1:-11:-1])
            print("current frame:", frame)
            print("True label:", true_label)
            # print("Top-10 probabilities/logits:", original_prob[-1:-11:-1])
            # print("Most unlikely classifications:", original_class[:10])
            # print("Most unlikely probabilities/logits:", original_prob[:10])
            if true_label != predicted_label:
                print("[WARNING] This image is classfied wrongly by the classifier! Skipping!")
                continue
            total += 1
            # set target class
            target_label = original_class[j]
            print('Target class: ', target_label)
            sys.stdout.flush()

            [L2_max,L1_max,Li_max,G2_max,G1_max,Gi_max,g_x0,pred] = clever_estimator.estimate(input[0], true_label, target_label, frame, Nsamp, Niters, 'l2', '', 1)
            # print("[STATS][L1] total = {}, id = {}, time = {:.3f}, true_class = {}, target_class = {}".format(total, idx, time.time() - timestart, true_label, target_label))


            # save to sampling results to matlab ;)
            mat_path = "{}/{}_{}/{}_{}_{}_{}_{}_order{}.mat".format('./lipschitz_mat', dataset, model_name, Nsamp, Niters,  true_label, target_label, 'relu', 1)
            save_dict = {'L2_max': L2_max, 'L1_max': L1_max, 'Li_max': Li_max, 'G2_max': G2_max, 'G1_max': G1_max, 'Gi_max': Gi_max, 'pred': pred, 'g_x0': g_x0, 'id': idx, 'true_label': true_label, 'target_label': target_label, 'path': mat_path}
            # sio.savemat(mat_path, save_dict)
            sys.stdout.flush()




            c_init = [0.1,1,5,10,20,50,100]

            # create thread pool
            nthreads = len(c_init) + 1
            print("using {} threads".format(nthreads))
            pool = Pool(processes = nthreads)
            # pool = Pool(1)
            # used for asynchronous plotting in background
            plot_res = None

            # get a list of all '.mat' files in folder
            # file_list = glob.glob('lipschitz_mat/'+dataset+'_'+model_name+'/**.mat', recursive = True)
            # # sort by image ID, then by information (least likely, random, top-2)
            # print(parse_filename(x)[2])
            # file_list = sorted(file_list, key = lambda x: (parse_filename(x)[2], parse_filename(x)[5]))


            # aggregate information for three different types: least, random and top2
            # each has three bounds: L1, L2, and Linf

            # nsamps, niters, true_id, true_label, target_label, img_info, activation, order = parse_filename(fname)

            # keys in mat:
            # ['Li_max', 'pred', 'G1_max', 'g_x0', 'path', 'info', 'G2_max', 'true_label', 'args', 'L1_max', 'Gi_max', 'L2_max', 'id', 'target_label']
            mat=save_dict
            order="1"
            if order == "1":
                G1_max = np.squeeze(mat['G1_max'])
                G2_max = np.squeeze(mat['G2_max'])
                Gi_max = np.squeeze(mat['Gi_max'])
            else:
                raise RuntimeError('!!! order is {}'.format(order))

            #
            # g_x0 = np.squeeze(mat['g_x0'])
            # target_label = np.squeeze(mat['target_label'])
            # true_id = np.squeeze(mat['id'])
            # true_label = np.squeeze(mat['true_label'])

            # get the filename (.mat)
            # get the model name (inception, cifar_2-layer)
            possible_names = ["mnist", "cifar", "mobilenet", "inception", "resnet"]
            model = dataset


            if order == "1":
                if norm == '1':
                    Est_G = get_lipschitz_estimate(Gi_max, pool, "Li", True)
                elif norm == '2':
                    Est_G = get_lipschitz_estimate(G2_max, pool, "L2", True)
                elif norm == 'i':
                    Est_G = get_lipschitz_estimate(G1_max, pool, "L1", True)

            # the estimated Lipschitz constant
            Lip_G = Est_G['Lips_est']

            # compute robustness bound
            if order == "1":
                bnd_L = g_x0 / Lip_G
            print(g_x0,'g_x0')
            print(Lip_G,'Lip_G')
            print(bnd_L,'bnd_L')

            temp={}
            temp['frame']=frame
            temp['truth']=true_label
            temp['target']=target_label
            temp['Lip']=Lip_G
            temp['bnd']=bnd_L
            bounds.append(temp)
            if Lip_G>largest_lip:
                largest_lip=Lip_G
            if bnd_L<smallest_adv:
                smallest_adv=bnd_L
            # original data_process mode
            #print('[STATS][L1] id = {}, true_label = {}, target_label = {}, info = {}, bnd_L1 = {:.5g}, bnd_L2 = {:.5g}, bnd_Li = {:.5g}'.format(true_id, true_label, target_label, img_info, bnd_L1, bnd_L2, bnd_Li))
            sys.stdout.flush()

            # shutdown thread pool
            pool.close()
            pool.join()

        frame += 1

    csv_columns=['frame','truth','target','Lip','bnd']
    csv_file="liplog.csv"
    with open(csv_file,'w') as f:
        writer=csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for data in bounds:
            writer.writerow(data)

    return bounds, time.time()-timestart

if __name__ == '__main__':
    bounds,runtime=collect_gradients('mnist', '256rnn71.hdf5', '2', 1)
    print(bounds)
