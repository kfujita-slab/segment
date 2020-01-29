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
    
# shows image on a named window. [img]: ndarray (RGB)
def show_img(img, window_name = "image", wait_time = 0.0):
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    time.sleep(wait_time)
    
# modified multi-process map function
# - <args>: [tuple], [list], or others
#   - length of [tuple]/[list] arguments must match <length>
#   - argument other than [tuple]/[list] is repeated <length> times
# - if <add_index> is True, a set of sequential numbers (0, 1, 2, ...)
#   is added to the head of <args>
def unpack_and_feed(pair):
    func, args = pair
    res = func(*args)
    res_q.put("prog")
    return res
def pool_map(func, length, add_index, *args):
    ex_args = [[i for i in range(length)]] if add_index else []
    ex_args.extend([(x if isinstance(x, (tuple, list)) else
                     [x for i in range(length)]) for x in args])
    pairs = [(func, [x[i] for x in ex_args]) for i in range(length)]
    pp    = Pool(processes = PROCESS_NUM)
    res   = pp.map_async(unpack_and_feed, pairs).get(3600)
    pp.close()
    pp.join()
    return res

# modified Queue/MPQueue which accepts multiple arguments for put()
# input data will be packed into tuple (even with 1 input data)
class multi_queue:
    def __init__(self, maxsize = 64, multiprocess = False):
        self.q = (MPQueue(maxsize = maxsize) if multiprocess else
                  Queue(maxsize = maxsize))
    def put(self, *data):
        if len(data) == 1:
            self.q.put((data[0], None))
        else:
            self.q.put(data)
    def get(self):                  
        while self.q.empty():
            time.sleep(0.1)
        return self.q.get()
                
#------------------------------------------------------------------------------
# type conversion functions
#------------------------------------------------------------------------------
# converts float to uint8
def float_to_uint8(img_in):
    img = img_in * 256.0 + 0.5
    return np.clip(img, 0., 255.).astype(np.uint8)

# converts uint8 to float32
def uint8_to_float(img_in):
    return img_in.astype(np.float32) / 256.0

# converts RGB color image to label image
def color_to_label(sv_in):
    lb = np.zeros(sv_in.shape[0:0+2], dtype = np.uint8)
    lb[np.where((128 <= sv_in[:,:,2]) & (sv_in[:,:,0] < 128))] = 1
    lb[np.where((128 <= sv_in[:,:,1]) & (sv_in[:,:,2] < 128))] = 2
    lb[np.where((128 <= sv_in[:,:,0]) & (sv_in[:,:,1] < 128))] = 3
    return lb

# converts label image to RGB color image
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
# adjust size of <img> to the specified size with scaling and trimming
# <scale_algo>: None, cv2.INTER_xxx
#------------------------------------------------------------------------------
def adjust_size(img, tgt_height, tgt_width, scale_algo, center = True):
    height, width = img.shape[0], img.shape[1]
    if scale_algo is not None:
        if (height * tgt_width) > (width * tgt_height):   # portrait
            height = (height * tgt_width) // width
            width  = tgt_width
        else:  # landscape
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
# loads image <initial + index>.<ext> from <path>
# - returns: np.ndarray
#   - mode == "label" : (height, width)      uint8
#   - mode == "train" : (height, width, RGB) uint8
#   - mode == "eval"  : (RGB, height, width) uint8
#------------------------------------------------------------------------------
def load_img(index, initial, dlen, path, ext, mode, height, width):
    # generates file name
    num = initial + index
    if dlen != 0:
        fname = "{0:s}{1:>0{dlen}d}.".format(path, num, dlen = dlen)
    else:
        fname = "{0:s}{1:d}.".format(path, num)
    # loads and adjust image size
    img = cv2.imread(fname + ext)
    if DEINTERLACE and (mode != "label"):
        img = cv2.resize(img[::2], (img.shape[1], img.shape[0]),
                         interpolation = cv2.INTER_LINEAR)
    img = adjust_size(img, height, width, cv2.INTER_NEAREST if mode == "label"
                      else cv2.INTER_LANCZOS4)
    if (np.amin(img) < 0) or (255 < np.amax(img)):
        error_exit("load_img: unexpected value detected")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    # reshaping
    if mode == "label":
        if len(img.shape) == 3:
            img = color_to_label(img)
        else:
            img = img.reshape(img.shape[0:0+2])
    else:
        if len(img.shape) != 3:
            error_exit("load_img: unsupported image format")
        if mode == "eval":
            img = img.transpose(2, 0, 1)
    return img

#------------------------------------------------------------------------------
# perform PCA for the input image
# - <img>: np.ndarray(height, width, RGB)
# - returns: (U, S)
#------------------------------------------------------------------------------
def perform_pca(img_in):
    img  = uint8_to_float(img_in.reshape(-1, 3))
    img *= np.sqrt(3.0 / np.sum(np.var(img, axis = 0)))
    U, S, V = np.linalg.svd(np.cov(img, rowvar = False))
    return (U, S)

#------------------------------------------------------------------------------
# generates [network input image] and [supervisory image]
#------------------------------------------------------------------------------
def generate_train_data(in_nw, in_lb, pc = None):
    tr, sv = in_nw.copy(), in_lb.copy()
    # random flip/rotation augmentation
    if RND_ROTATE:
        if random.randint(0, 1) == 1:
            tr, sv = tr[:, ::-1], sv[:, ::-1]
        tmat  = cv2.getRotationMatrix2D((tr.shape[1] // 2, tr.shape[0] // 2),
                                        random.random() * 360. , 1.0)
        tr = cv2.warpAffine(tr, tmat, (tr.shape[1], tr.shape[0]),
                            flags = cv2.INTER_LINEAR)
        sv = cv2.warpAffine(sv, tmat, (sv.shape[1], sv.shape[0]),
                            flags = cv2.INTER_NEAREST)
    # random cropping
    if RND_RESIZE:
        trim_size = int((2.0 / (random.random() + 1.0)) * INPUT_SIZE)
    else:
        trim_size = INPUT_SIZE
    v  = random.randint(0, in_lb.shape[0] - trim_size)
    h  = random.randint(0, in_lb.shape[1] - trim_size)
    tr = tr[v:v+trim_size, h:h+trim_size]
    sv = sv[v:v+trim_size, h:h+trim_size]
    # random resize augmentation
    if RND_RESIZE:
        tr = cv2.resize(tr, (INPUT_SIZE,)*2, interpolation=cv2.INTER_LANCZOS4)
        sv = cv2.resize(sv, (INPUT_SIZE,)*2, interpolation=cv2.INTER_NEAREST)
    # PCA color augmentation
    tr = uint8_to_float(tr)
    if COLOR_SHIFT:
        delta = np.dot(pc[0], pc[1] * np.random.randn(3) * 0.1)
        delta = delta.astype(np.float32).reshape((1, 1, 3))
        tr    = np.clip(tr + delta, 0.0, 1.0)
    # random erasing augmentataion
    if RND_ERASE and random.randint(0, 1) == 1:
        e_height = random.randint(INPUT_SIZE // 6, INPUT_SIZE // 2)
        e_width  = random.randint(INPUT_SIZE // 6, INPUT_SIZE // 2)
        ev = random.randint(0, INPUT_SIZE - e_height)
        eh = random.randint(0, INPUT_SIZE - e_width)
        tr[ev:ev+e_height, eh:eh+e_width] = random.random()  # [0.0, 1.0)
    if DEBUG:
        msg  = "scale: {0:4f}".format(INPUT_SIZE * 4.0 / (trim_size * 3.0))
        msg += "" if not COLOR_SHIFT else " delta: " + str(delta[0, 0])
        res_q.put("disp", msg)
        img = float_to_uint8(tr) // 2 + label_to_color(sv) // 2
        #show_img(float_to_uint8(tr), "train_batch", 0.0)
        #show_img(label_to_color(sv), "sv_batch",    1.0)
        show_img(img, "batch", 1.0)
    # reshaping and type conversion
    return tr.transpose(2, 0, 1), sv.astype(np.int32)

#------------------------------------------------------------------------------
# feeds training data
#------------------------------------------------------------------------------
def feed_data():
    # loads images and prepares dataset
    res_q.put("disp", "preparing training dataset")
    res_q.put("prog_set", TRAIN_PAIRS * 2)
    if RND_RESIZE:
        height, width = (IMG_HEIGHT * 4) // 3, (IMG_WIDTH * 4) // 3
    else:
        height, width = IMG_HEIGHT, IMG_WIDTH
    tr_nw = pool_map(load_img, TRAIN_PAIRS, True, 0, DIGIT_LEN,
                     INPUT_DIR, INPUT_EXT, "train", height, width)
    tr_lb = pool_map(load_img, TRAIN_PAIRS, True, 0, DIGIT_LEN,
                     LABEL_DIR, LABEL_EXT, "label", height, width)
    # prepares an iterator
    if COLOR_SHIFT:
        res_q.put("disp", "performing primary component analysis")
        res_q.put("prog_set", TRAIN_PAIRS)
        tr_pc   = pool_map(perform_pca, TRAIN_PAIRS, False, tr_nw)
        tr_data = [(tr_nw[i], tr_lb[i], tr_pc[i]) for i in range(TRAIN_PAIRS)] 
    else:
        tr_data = [(tr_nw[i], tr_lb[i]) for i in range(TRAIN_PAIRS)]
    serial_iter = SerialIterator(tr_data, BATCH_SIZE, shuffle = True)    
    data_q.put("prepare")
    for b in range(RESTART_POS, MAX_BATCH):
        batch = serial_iter.next()
        batch = [generate_train_data(*x) for x in batch]
        tb = np.concatenate([x[0][np.newaxis, :] for x in batch], axis = 0)
        sb = np.concatenate([x[1][np.newaxis, :] for x in batch], axis = 0)
        data_q.put("batch", (b + 1, tb.copy(), sb.copy()))
        if (b + 1) % EVAL_ITVL == 0 : 
            data_q.put("eval", b + 1)
    # termination
    data_q.put("end")
    
#------------------------------------------------------------------------------
# training
#------------------------------------------------------------------------------
def train_cnn():
    # model initialization
    if RESTART_POS == 0:
        model = SegNet()
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
    while True:
        msg, data = data_q.get()
        if msg == "end":
            eval_q.put("end")
            break
        elif msg == "prepare":
            # loads evaluation images
            res_q.put("disp", "loading evaluation input images")
            res_q.put("prog_set", EVAL_PAIRS)
            ev_nw = pool_map(load_img, EVAL_PAIRS, True,
                             TRAIN_PAIRS, DIGIT_LEN, INPUT_DIR, INPUT_EXT,
                             "eval",  IMG_HEIGHT, IMG_WIDTH)
            eval_q.put("prepare")
        # evaluation
        elif msg == "eval":
            eval_q.put("eval_start", data)
            res_q.put("disp", "evaluating @ batch {:d}".format(data))
            res_q.put("prog_set", len(ev_nw))
            # performs forward propagation and checks value distribution
            for i, nw in enumerate(ev_nw):
                lr = uint8_to_float(nw).reshape((1,) + nw.shape)
                if 0 <= GPU_ID:
                    lr = cuda.to_gpu(lr)
                with chainer.using_config("train", False):
                    with chainer.no_backprop_mode():
                        eval_q.put("eval_res", (i, model(lr, None)[0]))
                        res_q.put("prog")
            eval_q.put("eval_finish", model.copy().to_cpu())
        # training
        else:
            # receives batch data
            count, train_batch, sv_batch = data
            if 0 <= GPU_ID:
                train_batch = cuda.to_gpu(train_batch)
                sv_batch    = cuda.to_gpu(sv_batch)
            # update
            loss = model(train_batch, sv_batch)
            model.cleargrads()
            loss.backward()
            optim.update()
            eval_q.put("train", (count, float(cuda.to_cpu(loss.data))))

#------------------------------------------------------------------------------
# evaluation
#------------------------------------------------------------------------------
def eval_qual():
    max_count, max_accu = 0, 0.0
    while True:
        msg, data = eval_q.get()
        if msg == "end":
            res_q.put("end")
            break
        elif msg == "prepare":
            # loads evaluation supervisory images
            res_q.put("disp", "loading evaluation label images")
            res_q.put("prog_set", EVAL_PAIRS)
            ev_lb = pool_map(load_img, EVAL_PAIRS, True,
                             TRAIN_PAIRS, DIGIT_LEN, LABEL_DIR, LABEL_EXT,
                             "label", IMG_HEIGHT, IMG_WIDTH)
            res_q.put("disp", "saving evaluation label images")
            res_q.put("prog_set", EVAL_PAIRS)
            for i, x in enumerate(ev_lb):
                res_q.put("save_img", ("sv_img_{0:04d}.png".format(i),
                                       label_to_color(x)))
                res_q.put("prog", None)
        elif msg == "eval_start":
            train_count = data
            hit_pixs, all_pixs = [0] * CLASSES, [0] * CLASSES
            results, mean_accu, pixel_accu = [], [], []
        elif msg == "eval_res":
            i, res = data
            label, truth = np.argmax(res, axis = 0), ev_lb[i]
            pixel_accu.append(np.sum(label == truth) / float(truth.size))
            total_class_accu, total_classes = 0.0, 0
            for c in range(CLASSES):
                tgt = label[np.where(truth == c)]
                hp, ap = np.sum(tgt == c), tgt.size
                if ap != 0:
                    total_classes    += 1
                    total_class_accu += float(hp) / float(ap)
                hit_pixs[c] += hp
                all_pixs[c] += ap
            mean_accu.append(float(total_class_accu) / total_classes)
            results.append(label)
        elif msg == "eval_finish":
            total_mean_accu  = np.mean([float(hit_pixs[c]) / all_pixs[c]
                                        for c in range(CLASSES)])
            total_pixel_accu = float(sum(hit_pixs)) / sum(all_pixs)
            mean_accu  = [total_mean_accu]  + mean_accu
            pixel_accu = [total_pixel_accu] + pixel_accu
            res_q.put("save_accu", (train_count, mean_accu, pixel_accu))
            if ((total_mean_accu > max_accu) or
                (train_count % (FORCE_SAVE * EVAL_ITVL) == 0)):
                res_q.put("save_model", (train_count, data))
                for i, x in enumerate(results):
                    res_q.put("save_img", 
                              ("{0:08d}_{1:04d}.png".format(train_count, i),
                               label_to_color(x)))
            if total_mean_accu > max_accu:
                max_accu, max_count = total_mean_accu, train_count
            res_q.put("disp", "mean_accu: {0:f},".format(total_mean_accu) +
                      " max_mean_accu: {0:f}".format(max_accu) +
                      " @ batch {0:d}".format(max_count))
        elif msg == "train":
            res_q.put("train", data)
    
#------------------------------------------------------------------------------
# logger
#------------------------------------------------------------------------------
def hst_restart(path, target):
    new_data = ""
    if 0 < target:
        with open(path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            new_data += line
            if 0 <= line.find(" " + str(target) + " "):
                break
    with open(path, 'w') as fp:
        fp.write(new_data)

def hst_update(path, count, data):
    with open(path, 'a') as fp:
        # spaces before and after the count number MUST NOT be omitted!
        fp.write(" {0:8d}  ".format(count))
        fp.write(" ".join(["{0:f}".format(x) for x in data]) + "\n")
        
def log_result():
    # prepares quality list files
    if not DEBUG:
        hst_restart(RES_DIR + "qual_mean.txt",  RESTART_POS)
        hst_restart(RES_DIR + "qual_pixel.txt", RESTART_POS)
    while True:
        msg, data = res_q.get()
        if   msg == "disp":
            print("{0:s}\r- {1:s}".format(" " * 75, data))
        elif msg == "prog_set":
            finished, total = 0, data
        elif msg == "prog":
            finished += 1
            prog_bar(finished, total)
        elif msg == "save_accu":
            train_count, mean_accu, pixel_accu = data
            if not DEBUG:
                hst_update(RES_DIR + "qual_mean.txt",  train_count, mean_accu)
                hst_update(RES_DIR + "qual_pixel.txt", train_count, pixel_accu)
        elif msg == "save_model":
            train_count, model_data = data
            with open(RES_DIR +
                      "p_{0:08d}.pickle".format(train_count), 'wb') as fp:
                pickle.dump(model_data, fp)
        elif msg == "save_img":
            path, img = data
            if not DEBUG:
                cv2.imwrite(RES_DIR+path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        elif msg == "train":
            train_count, loss = data
            if train_count % 3 != 0:
                continue
            print("{0:s}\r".format(" " * 75) +
                  "- learning {:s} on GPU {:d}:".format(SYSTEM_NAME, GPU_ID) +
                  " (batch: {:d}, loss: {:f})\r".format(train_count, loss),
                  end = "")
            sys.stdout.flush()
        elif msg == "end":
            ctrl_q.put("end")
            break

#------------------------------------------------------------------------------
# confirmation
#------------------------------------------------------------------------------
def confirm():
    if 99999999 < MAX_BATCH:
        error_exit("The maximum value allowed for MAX_BATCH is 99999999")
    if ((IMG_HEIGHT < (INPUT_SIZE * 3) // 2) or
        (IMG_WIDTH  < (INPUT_SIZE * 3) // 2)):
        error_exit("IMG_HEIGHT / IMG_WIDTH is too small")
    if (IMG_HEIGHT % SIZE_UNIT != 0) or (IMG_WIDTH % SIZE_UNIT != 0):
        error_exit("IMG_HEIGHT / IMG_WIDTH must be the multiple of SIZE_UNIT")
    if (INPUT_SIZE % SIZE_UNIT != 0):
        error_exit("INPUT_SIZE must be the multiple of SIZE_UNIT")
    if DEBUG:
        return
    if not os.path.exists(RES_DIR):
        try:
            print("- creating output directory:")
            os.makedirs(RES_DIR)
        except IOError:
            error_exit("Failed to make the directory")
    if os.path.exists(RES_DIR + "qual_mean.txt"):
        if RESTART_POS == 0:
            print("A quality list already exists. Proceed? (yes/no)")
        else:
            print("Going to restart learning. Are you sure? (yes/no)")
        if input('>> ') != "yes" :
            sys.exit()
    elif RESTART_POS != 0:
        error_exit("The previous quality list doesn't exist")

#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
if __name__ == "__main__":
    confirm()
    data_q = multi_queue(multiprocess = False)
    eval_q = multi_queue(multiprocess = False)
    res_q  = multi_queue(multiprocess = True)
    ctrl_q = multi_queue(multiprocess = True)
    instances = (Thread( target = container, args = (feed_data,)),
                 Thread( target = container, args = (train_cnn,)),
                 Thread( target = container, args = (eval_qual,)),
                 Process(target = container, args = (log_result,)))
    try:
        for x in instances:
            x.daemon = True
            x.start()
        while True:
            msg, data = ctrl_q.get()
            if msg == "end": # successful termination
                for x in instances:
                    x.join()
                print("Learning " + SYSTEM_NAME + " finished successfully.")
                break
            else: # error detected
                sys.exit()
    except:  # exception detected
        print("\n\n! Something went wrong. Terminating...", file = sys.stderr)
        raise
