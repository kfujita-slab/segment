#-------------------------------------------------------------------------------
# <kf_learning.py>
#  - Learning program for CNN-based surgical image segmentation system
#  - Verified with:
#    - Python  2.7.6
#    - Chainer 5.3.0
#    - NumPy   1.16.2
#    - OpenCV  4.0.1.24
#-------------------------------------------------------------------------------
# Version 1.21-kf (Jan. 24, 2020)
#  - For the bachelor thesis of Kfujita-san
#-------------------------------------------------------------------------------
# (C) 2018-2020 Taito Manabe. All rights reserved.
#-------------------------------------------------------------------------------
# for compatibility with Python 3
from __future__        import division, print_function
from six.moves         import input, range, reduce
# others
import sys, os, time, random, pickle, cv2, chainer
import numpy               as np
import chainer.functions   as F
import chainer.links       as L
from chainer           import cuda, Variable, Chain, ChainList, optimizers
from chainer.iterators import SerialIterator
from itertools         import product
from threading         import Thread
from Queue             import Queue
from multiprocessing   import Process, Pool
from multiprocessing   import Queue as MPQueue

#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
# behavior
RESTART_POS = 0                   # if > 0, continues the previous learning
PROCESS_NUM = 4                   # num of processes for Pool
GPU_ID      = 0                   # CPU: -1, GPU: 0, 1, 2, ...
DEBUG       = False
SYSTEM_NAME = ""                  # to set automatically, leave this ""
RES_DIR     = "result/"        # directory where results are saved

# network configuration
REC_LEVEL   = 4                   # recursion level (L)
NW_CHS      = 20                  # num. of channels (i.e. depth)
CLASSES     = 4                   # number of classes (including background)
EXT_DEPTH   = 3                   # depth of Extract Unit (including conv3)
RDC_DEPTH   = 3                   # depth of Reduce Unit
MRG_DEPTH   = 3                   # depth of Merge Unit
NW_KSIZE    = 3                   # filter size
LRELU_SLOPE = 0.25                # slope (a) for Leaky ReLU

# learning
RND_ERASE   = False               # whether to use random erasing
RND_RESIZE  = True                # whether to use random resizing
RND_ROTATE  = True                # whether to use random rotation
COLOR_SHIFT = False               # whether to use PCA color augmentation
INPUT_SIZE  = 320                 # mini-batch image size
BATCH_SIZE  = 4                   # mini-batch size
EVAL_ITVL   = 1000                # evaluation interval
FORCE_SAVE  = 100                 # forced-save interval (* EVAL_ITVL)
MAX_BATCH   = 2000000             # learning limit (mini-batch) [1:99999999]

# dataset
IMG_HEIGHT  = 512                 # | Images will be resized to this size
IMG_WIDTH   = 640                 # | (must be >= 1.5 * INPUT_SIZE)
DEINTERLACE = True                # to apply EXTREMELY NAIVE deinterlace
INPUT_DIR   = "dataset/surgical/image/"
LABEL_DIR   = "dataset/surgical/label/"
INPUT_EXT   = "png"               # file extension of input RGB images
LABEL_EXT   = "png"               # file extension of label images
DIGIT_LEN   = 4                   # digit length
IMAGE_PAIRS = 183                 # num. of pairs in the dataset (train + eval)
TRAIN_PAIRS = 138                 # num. of pairs used for training
# network configuration
NW_FLT_SIZE = (3,3,3)           # filter size. each must be an odd number
NW_UNITS    = (3,12,12,12)
INT_BITW    = 5
FRAC_BITW   = 8                      # fractional part bit width
SLOPE_INV   = 4                      # 1 / (slope). should be 2^n. 0: ReLU

NW_DEPTH    = len(NW_FLT_SIZE)

# - both [INPUT_DIR] and [LABEL_DIR] must contain [DS_PAIRS] images
#   with file names following the format below:
#   - [0-filled sequential number].[INPUT_EXT/LABEL_EXT]

# following parameters are calculated automatically
if SYSTEM_NAME == "":
    SYSTEM_NAME = ("c{0:d}_l{1:d}_".format(NW_CHS, REC_LEVEL) +
                   ("e" if RND_ERASE   else "_") +
                   ("z" if RND_RESIZE  else "_") +
                   ("r" if RND_ROTATE  else "_") +
                   ("c" if COLOR_SHIFT else "_"))
if DEBUG:
    SYSTEM_NAME = "debug_" + SYSTEM_NAME
EVAL_PAIRS  = IMAGE_PAIRS - TRAIN_PAIRS
SIZE_UNIT   = 2 ** (REC_LEVEL)
FILE_DIR    = os.path.dirname(os.path.abspath(__file__)) + os.sep
INPUT_DIR   = FILE_DIR + INPUT_DIR
LABEL_DIR   = FILE_DIR + LABEL_DIR
RES_DIR     = FILE_DIR + RES_DIR + SYSTEM_NAME + os.sep

#------------------------------------------------------------------------------
# network definition
#------------------------------------------------------------------------------
# Extract Unit
class ExtUnit(ChainList):
    def __init__(self):  
        super(ExtUnit, self).__init__()
        for d in range(EXT_DEPTH):
            self.add_link(L.Convolution2D(3 if d == 0 else NW_CHS, NW_CHS,
                                          NW_KSIZE, pad = (NW_KSIZE - 1) // 2))
    def forward(self, x):
        for d in range(EXT_DEPTH):
            x = F.leaky_relu(self[d](x), slope = LRELU_SLOPE)
        return x

# Reduce Unit
class RdcUnit(ChainList):
    def __init__(self):  
        super(RdcUnit, self).__init__()
        for d in range(RDC_DEPTH):
            self.add_link(L.Convolution2D(NW_CHS, NW_CHS, NW_KSIZE,
                                          pad = (NW_KSIZE - 1) // 2))
    def forward(self, x):
        x = F.average_pooling_2d(x, 2)
        for d in range(RDC_DEPTH):
            x = F.leaky_relu(self[d](x), slope = LRELU_SLOPE)
        return x

# Merge Unit (a.k.a. Integrate Unit)
class MrgUnit(ChainList):
    def __init__(self):
        super(MrgUnit, self).__init__()
        for d in range(MRG_DEPTH):
            self.add_link(L.Convolution2D(NW_CHS * 2 if d == 0 else NW_CHS,
                                          NW_CHS, NW_KSIZE,
                                          pad = (NW_KSIZE - 1) // 2))
    def forward(self, x, y):
        x = F.concat((F.unpooling_2d(x, 2, cover_all = False), y), axis = 1)
        for d in range(MRG_DEPTH):
            x = F.leaky_relu(self[d](x), slope = LRELU_SLOPE)
        return x
# Top
class SegNet(Chain):
    def __init__(self):
        super(SegNet, self).__init__()
        with self.init_scope():
            self.ext = ExtUnit()
            self.rdc = RdcUnit()
            self.mrg = MrgUnit()
            self.lkh = L.Convolution2D(NW_CHS, CLASSES, 1)
    def forward(self, x_data, s_data):
        x, res = self.ext(Variable(x_data)), []
        # Reduce
        for d in range(REC_LEVEL):
            res.append(x)
            x = self.rdc(x)
        # Merge
        for y in reversed(res):
            x = self.mrg(x, y)
        # Conversion into likelihood map
        x = self.lkh(x)
        if chainer.config.train:
            return F.softmax_cross_entropy(x, Variable(s_data))
        else:
            return cuda.to_cpu(F.softmax(x).data)

#print(NW_DEPTH)

#------------------------------------------------------------------------------
# loads parameter file, and converts the parameters into fixed-point
#------------------------------------------------------------------------------
def to_fixed(msg, param, frac_len):
    pa = np.array(param)
    print("{0:s} vrange: [{1:f}, {2:f}]".format(msg,
                                                np.min(param), np.max(param)))
    pa = pa * (2.0 ** frac_len) + 0.5
    return (np.floor(pa)).astype(np.int64)

if not os.path.exists(RES_DIR):
    try:
        print("- creating output directory:")
        os.makedirs(RES_DIR)
    except IOError:
        error_exit("Failed to make the directory")
print("- loading parameter file:")
try:
    with open(FILE_DIR + "unit12_param.pickle", 'rb') as fp:
        model = pickle.load(fp)
except IOError:
    error_exit("! cannot open parameter file <param.pickle>")

#print(model.rdc.l0.W.data)

print("- converting parameters from float to fixed-point:")
flts   = [eval("model.ext[{0:d}].W.data".format(i+1-1)) for i in range(NW_DEPTH)]
biases = [eval("model.ext[{0:d}].b.data".format(i+1-1)) for i in range(NW_DEPTH)]
flts   = [to_fixed("    L{0:d} filter".format(i+1), flts[i], FRAC_BITW)
          for i in range(NW_DEPTH)]
biases = [to_fixed("    L{0:d} bias  ".format(i+1), biases[i], FRAC_BITW * 2)
          for i in range(NW_DEPTH)]

#------------------------------------------------------------------------------
# saves fixed-point parameters
#------------------------------------------------------------------------------
def fixed_to_str(data, bitw, msg):
    res    = ""
    data_r = np.ravel(data)
    for d in data_r:
        if d < 0 :
            d = d + (2 ** bitw)
        res += "{0:0>{bw}b}".format(d, bw = bitw)
    bitw   = len(res)
    res    = "{0:d}\'b".format(bitw) + res + ";\n"
    res    = "   localparam [0:{0:d}] {1:s} = ".format(bitw-1, msg) + res
    return res

with open(RES_DIR + "unit12_ext_fixed_param.txt", 'w') as fp:
    fp.write("")
with open(RES_DIR + "unit12_ext_fixed_param.txt", 'a') as fp:
    for i in xrange(len(flts)):
        fp.write(fixed_to_str(flts[i],   INT_BITW + FRAC_BITW,
                              "L{0:1d}_FLT".format(i+1)))
        fp.write(fixed_to_str(biases[i], INT_BITW + FRAC_BITW * 2,
                              "L{0:1d}_BIAS".format(i+1)))
