from __future__ import print_function
# from keras import backend as K
from tensorflow.keras import backend as K
import sys
import os

from NeuralNetwork import *
from DataSet import *
from DataCollection import *
from upperbound import upperbound
from lowerbound import lowerbound
from utils import save_diff,showImg
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# the first way of defining parameters

parser = argparse.ArgumentParser()
parser.add_argument('-b','--bound', default='ub', type=str)
parser.add_argument('-m','--mode', default='cooperative', type=str)
parser.add_argument('-i', default=32, type=int)
parser.add_argument('-utau', default=0.6, type=float)
parser.add_argument('-ltau', default=0.001, type=float)

args = parser.parse_args()

dataSetName = 'small'
bound = args.bound
gameType = args.mode
image_index = args.i
ueta = ('L2', 10)
utau = args.utau
leta = ('L2', 10)
ltau = args.ltau

# calling algorithms

if bound == 'ub':
    dc = DataCollection("%s_%s_%s_%s_%s_%s_%s" % (dataSetName, bound, utau, gameType, image_index, ueta[0], ueta[1]))
    dc.initialiseIndex(image_index)
    (elapsedTime, newConfident, percent, l2dist, l1dist, l0dist, maxFeatures) = upperbound(dataSetName, bound, utau,
                                                                                           gameType, image_index, ueta)
    dc.addRunningTime(elapsedTime)
    dc.addConfidence(newConfident)
    dc.addManipulationPercentage(percent)
    dc.addl2Distance(l2dist)
    dc.addl1Distance(l1dist)
    dc.addl0Distance(l0dist)
    dc.addMaxFeatures(maxFeatures)

elif bound == 'lb':
    dc = DataCollection("%s_%s_%s_%s_%s_%s_%s" % (dataSetName, bound, ltau, gameType, image_index, leta[0], leta[1]))
    dc.initialiseIndex(image_index)
    lowerbound(dataSetName, image_index, gameType, leta, ltau)

else:
    print("lower bound algorithm is developing...")
    exit

dc.provideDetails()
dc.summarise()
dc.close()

K.clear_session()
