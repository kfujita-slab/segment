#-------------------------------------------------------------------------------
# <learning.py>
#  - Learning program for CNN-based surgical image segmentation system
#  - Verified with:
#    - Python   2.7.6
#    - Chainer  5.3.0
#    - NumPy    1.16.2
#    - OpenCV   4.0.0.21
#-------------------------------------------------------------------------------
# Version 1.14 (Mar. 15, 2019)
#  - Compatible with Chainer 5.3.0
#  - Replaced skimage and Pillow with OpenCV
#  - Changed so that input images are resized automatically
#    to <IMG_HEIGHT> and <IMG_WIDTH>, using scaling and trimming
#  - Added color shift (Hue, Saturation, Luminance) augmentation
#  - Reverted random rotation augmentation back to random flip
#  - Removed value distribution check in the evaluation process
#  - Other minor refinements
#-------------------------------------------------------------------------------
# (C) 2018-2019 Taito Manabe. All rights reserved.
#-------------------------------------------------------------------------------
# for compatibility with Python 3
from __future__      import division, print_function
from six.moves       import input, range, reduce
# others
import sys, os, time, copy, random, pickle, cv2, chainer
import numpy             as np
import chainer.functions as F
import chainer.links     as L
from itertools       import product
from chainer         import cuda, Variable, Chain, ChainList, optimizers
from threading       import Thread
from Queue           import Queue
from multiprocessing import Process, Pool
from multiprocessing import Queue as MPQueue

#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
# behavior
RESTART_POS = 0                   # if > 0, restarts the previous learning
PROCESS_NUM = 4                   # num of processes for Pool
GPU_ID      = 1                   # CPU: -1, GPU: 0, 1, 2, ...
RES_DIR     = "result/ver_1_14/"  # directory where results are saved

# system configuration
SYSTEM_NAME = "hsv_none"
NW_SCALE    = 5                   # num. of image resolutions in the network
NW_DEPTHS   = (3, 3, 3)           # Ext, Rdc, Itg
NW_UNITS    = 16                  # num. of output units each layer has
NW_KSIZE    = 3                   # filter size
NW_USE_RES  = True                # to use residual block or not

# learning
COLOR_SHIFT = False                # to use color shift augmentation or not
RND_ERASE   = False               # to use random erasing or not
INPUT_SIZE  = 512                 # size of images in mini-batches
BATCH_SIZE  = 1                   # size of mini-batches
LRELU_SLOPE = 0.25                # leaky ReLU slope. 0 is the same as ReLU
EVAL_ITVL   = 2500                # evaluation interval
FORCE_SAVE  = 80                  # forced-save interval (* EVAL_ITVL)
MAX_BATCH   = 10000000            # learning limit (mini-batch) [1:99999999]

# dataset
IMG_HEIGHT  = 512                 # | Images will be resized to this size
IMG_WIDTH   = 640                 # | (must be >= INPUT_SIZE)
CLASSES     = 4                   # number of classes (including background)
IMG_INIT    = 0                    
IMG_NUM     = 52
EVAL_INIT   = 52
EVAL_NUM    = 16                  # [1:10000]
DIGIT_LEN   = 4                   # digit length. 0: no zero fill
IMG_DIR     = "dataset/surgical/"
EVAL_DIR    = "dataset/surgical/"
INPUT_DIR   = "image/"
LABEL_DIR   = "label/"
INPUT_EXT   = "png"
LABEL_EXT   = "png"
# each of IMG_DIR/EVAL_DIR must contain two sub directories:
#   - INPUT_DIR : input images ([serial number].[INPUT_EXT])
#   - LABEL_DIR : label images ([serial number].[LABEL_EXT])
# examples of [serial number] are as follows:
#   - IMG_INIT == 3, IMG_NUM == 5, DIGIT_LEN == 4:
#       0003, 0004, 0005, 0006, 0007
#   - IMG_INIT == 7, IMG_NUM == 10, DIGIT_LEN == 0 (no zero fill):
#       7, 8, 9, 10, 11, 12, 13, 14, 15, 16

# following parameters are calculated automatically
PAD_WIDTH   = (NW_KSIZE - 1) // 2
SIZE_UNIT   = 2 ** (NW_SCALE - 1)
IMG_HEIGHT  = (IMG_HEIGHT // SIZE_UNIT) * SIZE_UNIT
IMG_WIDTH   = (IMG_WIDTH  // SIZE_UNIT) * SIZE_UNIT
FILE_DIR    = os.path.dirname(os.path.abspath(__file__)) + os.sep
IMG_DIR     = FILE_DIR + IMG_DIR
EVAL_DIR    = FILE_DIR + EVAL_DIR
RES_DIR     = FILE_DIR + RES_DIR + SYSTEM_NAME + os.sep

#------------------------------------------------------------------------------
# network definition
#------------------------------------------------------------------------------
# ExtNet
class extnet(ChainList):
    def __init__(self):
        super(extnet, self).__init__()
        for d in range(NW_DEPTHS[0]):
            self.add_link(L.Convolution2D(3 if d == 0 else NW_UNITS,
                                          NW_UNITS, NW_KSIZE, pad = PAD_WIDTH))
    def forward(self, x):
        for d in range(NW_DEPTHS[0]):
            x = self[d](x)
            x = F.leaky_relu(x, slope = LRELU_SLOPE)
            if NW_USE_RES and d == 0:
                skip = x
        if NW_USE_RES:
            x = x + skip
        return x

# RdcNet
class rdcnet(ChainList):
    def __init__(self):
        super(rdcnet, self).__init__()
        for d in range(NW_DEPTHS[1]):
            self.add_link(L.Convolution2D(NW_UNITS, NW_UNITS,
                                          NW_KSIZE, pad = PAD_WIDTH))
    def forward(self, x):
        if NW_USE_RES:
            skip = x
        for d in range(NW_DEPTHS[1]):
            x = self[d](x)
            x = F.leaky_relu(x, slope = LRELU_SLOPE)
        if NW_USE_RES:
            x = x + skip
        return x    
    
# ItgNet
class itgnet(ChainList):
    def __init__(self):
        super(itgnet, self).__init__()
        a = NW_UNITS * 2
        for d in range(NW_DEPTHS[2]):
            b = (NW_UNITS * (NW_DEPTHS[2] * 2 - d - 1)) // NW_DEPTHS[2]
            self.add_link(L.Convolution2D(a, b, NW_KSIZE, pad = PAD_WIDTH))
            a = b
    def forward(self, x, y):
        x = F.concat((F.unpooling_2d(x, 2, cover_all = False), y), axis = 1)
        for d in range(NW_DEPTHS[2]):
            x = self[d](x)
            x = F.leaky_relu(x, slope = LRELU_SLOPE)
        if NW_USE_RES:
            x = x + y
        return x    

# Top network
class segnet(Chain):
    def __init__(self):
        super(segnet, self).__init__()
        with self.init_scope():
            self.ext = extnet()
            self.rdc = rdcnet()
            self.itg = itgnet()
            self.lkh = L.Convolution2D(NW_UNITS, CLASSES, 1, pad = 0)
    def forward(self, xb, sb):
        # ExtNet
        x = self.ext(Variable(xb))
        imgs = []
        imgs.append(x)
        # RdcNet
        for s in range(NW_SCALE - 1):
            x = self.rdc(F.max_pooling_2d(x, 2))
            if s != NW_SCALE - 2:
                imgs.append(x)
        # ItgNet
        for y in reversed(imgs):
            x = self.itg(x, y)
        x = self.lkh(x)
        if chainer.config.train:
            return F.softmax_cross_entropy(x, Variable(sb))
        else:
            return cuda.to_cpu(F.softmax(x).data)

#------------------------------------------------------------------------------
# Queues used by Threads/Processes to communicate with each other
#------------------------------------------------------------------------------
class new_queue:
    def __init__(self, maxsize = 64, multiprocess = False):
        self.q = (MPQueue(maxsize = maxsize) if multiprocess else
                  Queue(maxsize = maxsize))
    def put(self, data1, data2 = None):
        self.q.put((data1, data2))
    def get(self):                  
        while self.q.empty():
            time.sleep(0.1)
        return self.q.get()
    def cancel_join_thread(self):
        self.q.cancel_join_thread()

data_q = new_queue(multiprocess = False)
res_q  = new_queue(multiprocess = True)
save_q = new_queue(multiprocess = True)
ctrl_q = new_queue(multiprocess = True)
                
#------------------------------------------------------------------------------
# error handling functions
#------------------------------------------------------------------------------
def error_exit(msg):
    print("! " + msg, file = sys.stderr)
    ctrl_q.put("exit")
    sys.exit()
def container(func):
    try:
        func()
    except:  # catch exception (including SystemExit)
        ctrl_q.put("exit")
        raise

#------------------------------------------------------------------------------
# confirmation
#------------------------------------------------------------------------------
if 99999999 < MAX_BATCH:
    error_exit("The maximum value allowed for MAX_BATCH is 99999999")
if (IMG_HEIGHT < INPUT_SIZE) or (IMG_WIDTH < INPUT_SIZE):
    error_exit("IMG_HEIGHT and IMG_WIDTH must not be smaller than INPUT_SIZE")
if not os.path.exists(RES_DIR):
    try:
        print("- creating output directory:")
        os.makedirs(RES_DIR)
    except IOError:
        error_exit("Failed to make the directory")
if os.path.exists(RES_DIR + "qual_all.txt"):
    if RESTART_POS == 0:
        print("A quality list already exists. Proceed? (yes/no)")
    else:
        print("Going to restart learning. Are you sure? (yes/no)")
    if input('>> ') != "yes" :
        sys.exit()
elif RESTART_POS != 0:
    error_exit("The previous quality list doesn't exist")

#------------------------------------------------------------------------------
# useful functions
#------------------------------------------------------------------------------
# displays progress bar
def prog_bar(finished, total, width = 60, autohide = True):
    progress  = float(finished) / float(total) if (0 < total) else 0
    finished  = int(width * progress)
    remaining = width - finished
    print("  |{0:s}{1:s}| ".format("=" * finished, " " * remaining) +
          " {0:3.1f}%\r".format(progress * 100.0), end = "")
    if autohide and remaining == 0:
        print("{0:s}\r".format(" " * (width + 12)), end = "")
    sys.stdout.flush()
# shows image on a named window
def show_img(img, window_name = "image", wait_time = 0.0):
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    time.sleep(wait_time)
# conversion between float and uint8
def float_to_uint8(img_in):
    img = img_in * 256.0 + 0.5
    return np.clip(img, 0., 255.).astype(np.uint8)
def uint8_to_float(img_in):
    return img_in.astype(np.float32) / 256.0
    
#------------------------------------------------------------------------------
# multi-process map function
# args: [tuple], [list], or others
# length of [tuple] or [list] arguments in <args> must match <length>
#------------------------------------------------------------------------------
def pool_map(func, length, index = False, *args):
    ex_args  = [(x if isinstance(x, (tuple, list)) else
                 [x for i in range(length)]) for x in args]
    if index:
        ex_args = [[i for i in range(length)]] + ex_args
    argpairs = [[x[i] for x in ex_args] for i in range(length)]
    pp  = Pool(processes = PROCESS_NUM)
    res = pp.map_async(func, argpairs).get(3600)
    pp.close()
    pp.join()
    return res
    
#------------------------------------------------------------------------------
# conversion between the colored segmentation image (RGB) and label data
#------------------------------------------------------------------------------
def color_to_label(sv_in):
    lb = np.zeros(sv_in.shape[0:0+2], dtype = np.uint8)
    lb[np.where((128 <= sv_in[:,:,2]) & (sv_in[:,:,0] < 128))] = 1
    lb[np.where((128 <= sv_in[:,:,1]) & (sv_in[:,:,2] < 128))] = 2
    lb[np.where((128 <= sv_in[:,:,0]) & (sv_in[:,:,1] < 128))] = 3
    return lb

def label_to_color(lb_in):
    res   = np.zeros(lb_in.shape + (3,), dtype = np.uint8)
    tgt   = np.where(lb_in != 0)
    label = lb_in.astype(np.float32)
    hue   = ((CLASSES - 1. - label) / float(CLASSES - 2)) * 0.42
    res[:,:,0][tgt]     = float_to_uint8(hue)[tgt]
    res[:,:,1:1+2][tgt] = 255
    res   = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    return res

#------------------------------------------------------------------------------
# adjust size of <img> to the specified size with scaling and trimming
# <scale_algo>: None, cv2.INTER_xxx
#------------------------------------------------------------------------------
def adjust_size(img, tgt_height, tgt_width, scale_algo, center = True):
    height, width = img.shape[0], img.shape[1]
    if scale_algo is not None:
        if (height * tgt_width) > (width * tgt_height):   # portrait
            height = (height * tgt_width) // width
            width  = tgt_width
        else:
            width  = (width * tgt_height) // height
            height = tgt_height
        res = cv2.resize(img, (width, height), interpolation = scale_algo)
    else:
        res = img
    if center:
        trim_height = (height - tgt_height) // 2
        trim_width  = (width  - tgt_width)  // 2
        res = res[trim_height:trim_height+tgt_height,
                  trim_width:trim_width+tgt_width]
    else:
        res = res[0:tgt_height, 0:tgt_width]
    return res.copy()

#------------------------------------------------------------------------------
# loads <num> images xxx.png from <path>.
# xxx is a <digit_len>-digit serial number.
# example: (000, 001, 002, ..., 029)  (<num> = 30, <digit_len> = 3)
# <mode>:
#   - "train": np.uint8 (height, width, channel(HSV or RGB))  * H: [0, 179]
#   - "eval" : np.uint8 (channel(RGB), height, width)
#   - "label": np.uint8 (height, width)
#------------------------------------------------------------------------------
def load_img(pair):
    index, path, ext, initial, dlen, mode = pair
    # generates file name
    num = initial + index
    if dlen != 0:
        fname = "{0:s}{1:>0{dlen}d}.".format(path, num, dlen = dlen)
    else:
        fname = "{0:s}{1:d}.".format(path, num)
    # loads and adjust image size
    img = adjust_size(cv2.imread(fname + ext), IMG_HEIGHT, IMG_WIDTH,
                      cv2.INTER_NEAREST if mode == "label" else
                      cv2.INTER_LANCZOS4)
    if (np.amin(img) < 0) or (255 < np.amax(img)):
        error_exit("load_img: unexpected value detected")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    res_q.put("prog")
    if mode == "label":
        if len(img.shape) == 3 and 3 <= img.shape[2]:
            img = color_to_label(img)
        else:
            img = img.reshape(img.shape[0:0+2])
    else:
        if len(img.shape) != 3 or img.shape[2] < 3:
            error_exit("load_img: unsupported image format")
        if mode == "train":
            if COLOR_SHIFT:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif mode == "eval":
            img = img[:, :, 0:0+3].transpose((2, 0, 1))
    res_q.put("prog")
    return img

#------------------------------------------------------------------------------
# calculates the quality of segmentation
#------------------------------------------------------------------------------
def eval_qual(pair):
    number, res, truth = pair
    res   = res[0]
    label = np.argmax(res, axis = 0)
    hit_pixs, total_pixs = [], []
    for c in range(CLASSES):
        tgt = label[np.where(truth == c)]
        hit_pixs.append(np.sum(tgt == c))
        total_pixs.append(tgt.size)
    res_q.put("prog")
    res_q.put("eval_data", (number, label, hit_pixs, total_pixs))

def eval_qual_all(nw_imgs, lb_imgs, model):
    # performs forward propagation and checks value distribution
    res = []
    for nw in nw_imgs:
        lr = uint8_to_float(nw).reshape((1,) + nw.shape)
        if 0 <= GPU_ID:
            lr = cuda.to_gpu(lr)
        with chainer.using_config("train", False):
            with chainer.no_backprop_mode():
                r = model(lr, None)
        res.append(r)
        res_q.put("prog", None)
    # evaluation and queuing
    pool_map(eval_qual, len(res), True, res, lb_imgs)

#------------------------------------------------------------------------------
# generates [network input image] and [supervisory image]
#------------------------------------------------------------------------------
def generate_train_data(in_nw, in_lb):
    # clip
    v  = random.randint(0, in_lb.shape[0] - INPUT_SIZE)
    h  = random.randint(0, in_lb.shape[1] - INPUT_SIZE)
    tr = in_nw[v:v+INPUT_SIZE, h:h+INPUT_SIZE].copy()
    sv = in_lb[v:v+INPUT_SIZE, h:h+INPUT_SIZE].copy()
    # random flips (augmentation)
    if random.randint(0, 1) == 1:
        tr, sv = tr[::-1, :], sv[::-1, :]
    if random.randint(0, 1) == 1:
        tr, sv = tr[:, ::-1], sv[:, ::-1]
    # random hue/saturation/luminance color shift (augmentation)
    if COLOR_SHIFT:
        tr = tr.astype(np.int16)
        tr[:,:,0] = (tr[:,:,0] + 180 + random.randint(-18, 18)) % 180 # H
        tr[:,:,1] = (tr[:,:,1] * random.randint(6, 16)) // 16         # S
        tr = np.clip(tr, 0, 255).astype(np.uint8)
        tr = cv2.cvtColor(tr, cv2.COLOR_HSV2RGB)
        tr = (tr.astype(np.int16) * random.randint(8, 24)) // 16      # Y
        tr = np.clip(tr, 0, 255).astype(np.uint8)
    # random erasing (augmentation). label data won't be affected
    if RND_ERASE and random.randint(0, 1) == 1:
        e_height = random.randint(INPUT_SIZE // 6, INPUT_SIZE // 2)
        e_width  = random.randint(INPUT_SIZE // 6, INPUT_SIZE // 2)
        ev = random.randint(0, INPUT_SIZE - e_height)
        eh = random.randint(0, INPUT_SIZE - e_width)
        tr[ev:ev+e_height, eh:eh+e_width] = random.randint(0, 255)
    # type conversion
    tr, sv = uint8_to_float(tr.transpose((2, 0, 1))), sv.astype(np.int32)
    return tr, sv

#------------------------------------------------------------------------------
# feeds training data
#------------------------------------------------------------------------------
def feed_train_data():
    # loads images and prepares dataset
    res_q.put("disp", "preparing training dataset")
    res_q.put("prog_set", IMG_NUM * 4)
    tr_nw = pool_map(load_img, IMG_NUM, True, IMG_DIR + INPUT_DIR,
                     INPUT_EXT, IMG_INIT, DIGIT_LEN, "train")
    tr_lb = pool_map(load_img, IMG_NUM, True, IMG_DIR + LABEL_DIR,
                     LABEL_EXT, IMG_INIT, DIGIT_LEN, "label")
    data_q.put("prepare")
    # preparation
    tb = np.ndarray((BATCH_SIZE, 3) + (INPUT_SIZE,) * 2, dtype = np.float32)
    sb = np.ndarray((BATCH_SIZE,)   + (INPUT_SIZE,) * 2, dtype = np.int32)
    count = 0
    # train data feeding loop
    for batch in range(MAX_BATCH):
        for i in range(BATCH_SIZE):
            if count == 0:
                order = np.random.permutation(len(tr_nw))
            tb[i], sb[i] = generate_train_data(tr_nw[order[count]],
                                               tr_lb[order[count]])
            count = (count + 1) % len(tr_nw)
        data_q.put("batch", (tb.copy(), sb.copy()))
        if (batch + 1) % EVAL_ITVL == 0 : 
            data_q.put("eval")
    # termination
    data_q.put("end")
def train_data_feeder():
    container(feed_train_data)
    
#------------------------------------------------------------------------------
# training
#------------------------------------------------------------------------------
def train_cnn():
    # model initialization
    if RESTART_POS == 0:
        model = segnet()
    else:
        param_path = RES_DIR + "p_{0:08d}.pickle".format(RESTART_POS)
        try:
            with open(param_path, 'rb') as fp:
                model = pickle.load(fp)
        except IOError:
            error_exit("train_cnn: cannot load a parameter file " + param_path)
    if 0 <= GPU_ID:
        cuda.check_cuda_available()
        cuda.get_device(GPU_ID).use()
        model.to_gpu()
    optim = optimizers.Adam().setup(model)
    # training loop
    while True:
        msg, data = data_q.get()
        if msg == "end":
            res_q.put("end")
            break
        elif msg == "prepare":
            # loads evaluation images
            res_q.put("disp", "preparing evaluation dataset")
            res_q.put("prog_set", EVAL_NUM * 4)
            ev_nw = pool_map(load_img, EVAL_NUM, True, EVAL_DIR + INPUT_DIR,
                             INPUT_EXT, EVAL_INIT, DIGIT_LEN, "eval")
            ev_lb = pool_map(load_img, EVAL_NUM, True, EVAL_DIR + LABEL_DIR,
                             LABEL_EXT, EVAL_INIT, DIGIT_LEN, "label")
            res_q.put("disp", "saving supervisory label data")
            res_q.put("prog_set", EVAL_NUM)
            for i, x in enumerate(ev_lb):
                save_q.put(RES_DIR + "sv_img_{0:04d}.png".format(i),
                           label_to_color(x))
                res_q.put("prog", None)
        # evaluation
        elif msg == "eval":
            res_q.put("eval_start")
            res_q.put("prog_set", len(ev_nw) * 2)
            eval_qual_all(ev_nw, ev_lb, model)
            res_q.put("eval_param", model.copy().to_cpu())
        # training
        else:
            # receives batch data
            train_batch, sv_batch = data
            if 0 <= GPU_ID:
                train_batch = cuda.to_gpu(train_batch)
                sv_batch    = cuda.to_gpu(sv_batch)
            # update
            loss = model(train_batch, sv_batch)
            model.cleargrads()
            loss.backward()
            optim.update()
            res_q.put("train", float(cuda.to_cpu(loss.data)))
def cnn_trainer():
    container(train_cnn)
        
#------------------------------------------------------------------------------
# logger
#------------------------------------------------------------------------------
def hst_restart(path, target, length):
    if length <= 0:
        error_exit("hst_restart: <length> must be positive")
    new_data = ""
    if 0 < target:
        with open(path, 'r') as fp:
            lines = fp.readlines()
        count = -1
        for line in lines:
            if count == 0:
                break
            elif 0 < count:
                count -= 1
            elif 0 <= line.find(" " + str(target) + " "):
                count = length - 1
            new_data += line
    with open(path, 'w') as fp:
        fp.write(new_data)

def hst_update(path, count, msg):
    with open(path, 'a') as fp:
        # spaces before and after the count number MUST NOT be omitted!
        fp.write(" {0:8d}  {1:s}".format(count, msg))
        
def log_result():
    # prepares quality list files
    for i in range(EVAL_NUM):
        hst_restart(RES_DIR + "qual_{0:04d}.txt".format(i), RESTART_POS, 1)
    hst_restart(RES_DIR + "qual_all.txt",  RESTART_POS, 1)
    # preparation
    zero_list     = [0 for c in range(CLASSES)]
    train_count   = RESTART_POS
    max_mean_accu = 0.0
    max_count     = 0
    evaluating, model_ready, exiting = (False,) * 3
    results, hit_pixs, total_pixs    = [], zero_list, zero_list
    # logging loop
    while True:
        msg, data = res_q.get()
        if msg == "disp":
            print("{0:s}\r- {1:s}:".format(" " * 75, data))
        elif msg == "prog_set":
            finished = 0
            total    = data
        elif msg == "prog":
            finished += 1
            prog_bar(finished, total)
        elif msg == "eval_start":
            evaluating = True
            eval_count = train_count
            print(" " *75 + "\r- evaluating @ batch {:d}:".format(eval_count))
        elif msg == "eval_data":
            n, label, hp, tp = data
            aa, acount = 0.0, 0
            for c in range(CLASSES):
                if tp[c] != 0:
                    aa     += float(hp[c]) / float(tp[c])
                    acount += 1
            aa /= float(acount)
            pa  = float(sum(hp)) / float(sum(tp))
            hst_update(RES_DIR + "qual_{0:04d}.txt".format(n), eval_count,
                       "{0:f} {1:f}\n".format(aa, pa))
            results.append((n, label))
            hit_pixs   = [hit_pixs[c]   + hp[c] for c in range(CLASSES)]
            total_pixs = [total_pixs[c] + tp[c] for c in range(CLASSES)]
        elif msg == "eval_param":
            model_data  = data
            model_ready = True
        elif msg == "train":
            train_count += 1
            if train_count % 3 != 0:
                continue
            print("{0:s}\r".format(" " * 75) +
                  "- learning {:s} on GPU {:d}:".format(SYSTEM_NAME, GPU_ID) +
                  " (batch: {:d}, loss: {:f})\r".format(train_count, data),
                  end = "")
            sys.stdout.flush()
        elif msg == "end":
            exiting = True
        # finishes evaluation
        if evaluating and model_ready and len(results) == EVAL_NUM:
            mean_accu = sum([float(hit_pixs[c]) / float(total_pixs[c])
                             for c in range(CLASSES)]) / float(CLASSES)
            pix_accu = float(sum(hit_pixs)) / float(sum(total_pixs))
            # updates quality list file
            hst_update(RES_DIR + "qual_all.txt", eval_count,
                       "{0:f} {1:f}\n".format(mean_accu, pix_accu))
            if (max_mean_accu < mean_accu or
                eval_count % (EVAL_ITVL * FORCE_SAVE) == 0):
                # update the maximum hit ratio
                if(max_mean_accu < mean_accu) :
                    max_mean_accu, max_count = mean_accu, eval_count
                # save a processed image and current parameters
                for i in range(EVAL_NUM):
                    n, res = results[i]
                    img_name = "{:08d}_{:04d}.png".format(eval_count, n)
                    save_q.put(RES_DIR + img_name, label_to_color(res))
                    prog_bar(i+1, EVAL_NUM)
                with open(RES_DIR +
                          "p_{:08d}.pickle".format(eval_count), 'wb') as fp:
                    pickle.dump(model_data, fp)
            # print ssim
            print("{0:s}\r    mean_accu: {1:f},".format(" " * 75, mean_accu) +
                  " max_mean_accu: {0:f} @ batch {1:d}".format(max_mean_accu,
                                                               max_count))
            # initialize
            evaluating, model_ready = False, False
            results, hit_pixs, total_pixs = [], zero_list, zero_list
        # if training and evaluation have finished
        if exiting and not evaluating:
            save_q.put("end")
            break
def result_logger():
    container(log_result)

#------------------------------------------------------------------------------
# image save
#------------------------------------------------------------------------------
def save_img():
    while True:
        path, img = save_q.get()
        if path == "end":
            ctrl_q.put("end")
            break
        else:
            try:
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            except IOError:
                error_exit("img_saver: cannot write {0:s}").format(path)
def img_saver():
    container(save_img)

#------------------------------------------------------------------------------
# starts threads / processes
#------------------------------------------------------------------------------
instances = (Thread(target = train_data_feeder),
             Thread(target = cnn_trainer),
             Process(target = result_logger),
             Process(target = img_saver))
def cleanup():
    print("\n\n! Something went wrong. Terminating...", file = sys.stderr)
    res_q.cancel_join_thread()
    save_q.cancel_join_thread()
    ctrl_q.cancel_join_thread()
try:
    for x in instances:
        x.daemon = True
        x.start()
    while True:
        msg, data = ctrl_q.get()
        if msg == "end": # successful termination
            for x in instances:
                x.join()
            print("Learning {0:s} finished successfully.".format(SYSTEM_NAME))
            break
        else: # error detected
            sys.exit()
except:  # exception detected
    cleanup()
    raise
