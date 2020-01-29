#-------------------------------------------------------------------------------
# <learning.py>
#  - Learning program for CNN-based super-resolution system
#  - Compatible configurations:
#    - "pe" : pre-enlargement SR method used in SRCNN
#    - "sp" : sub-pixel SR method used in ESPCN
#    - "fp" : flip SR method using 4 combinations of H/V flips
#  - Verified with:
#    - Python       2.7.6
#    - Chainer      5.3.0
#    - NumPy        1.16.2
#    - OpenCV       4.0.0.21
#    - scikit-image 0.14.3
#-------------------------------------------------------------------------------
# Version 6.01 (June 27, 2019)
#  - Removed batch renormalization since it seems not to be effective
#    for super-resolution CNNs
#  - Improved HLS color shift algorithm to mitigate abnormal colors
#-------------------------------------------------------------------------------
# (C) 2015-2019 Taito Manabe. All rights reserved.
#-------------------------------------------------------------------------------
# for compatibility with Python 3
from __future__        import division, print_function
from six.moves         import input, range, reduce
# others
import sys, os, time, random, pickle, cv2, chainer
import numpy               as np
import skimage.measure     as sk_ms
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
DEBUG       = False               # whether to show training image pairs
RES_DIR     = "result/ver_6_01/"  # directory where results are saved

# network configuration
SR_METHOD   = "fp"                # "pe", "sp", "fp"
NW_KSIZE    = 3                   # filter size
NW_DEPTH    = 5                   # network depth
NW_UNITS    = 64                  # num. of units in each layer
LRELU_SLOPE = 0.25                # slope (a) for Leaky ReLU

# learning
COLOR_SHIFT = True                # whether to use HLS color shift
JPEG_COMP   = True               # whether to compress input images
INPUT_SIZE  = 256                 # mini-batch image size. must be even
BATCH_SIZE  = 4                   # mini-batch size
EVAL_ITVL   = 2500                # evaluation interval
FORCE_SAVE  = 40                  # forced-save interval (* EVAL_ITVL)
MAX_BATCH   = 10000000            # learning limit (mini-batch) [1:99999999]

# dataset
TRAIN_NUM   = 300                 # num. of images for training
EVAL_NUM    = 30                  # num. of images for evaluation
DIGIT_LEN   = 3                   # 0: no 0-fill
TRAIN_DIR   = "dataset/train/"    # | place training/evaluation images here:
EVAL_DIR    = "dataset/eval/"     # |  <000.png>, <001.png>, <002.png>, ...

# following parameters are calculated automatically
SYSTEM_NAME = ("_".join([SR_METHOD, str(NW_DEPTH), str(NW_UNITS)]) +
               ("_dn" if JPEG_COMP else ""))
FILE_DIR    = os.path.dirname(os.path.abspath(__file__)) + os.sep
TRAIN_DIR   = FILE_DIR + TRAIN_DIR
EVAL_DIR    = FILE_DIR + EVAL_DIR
RES_DIR     = FILE_DIR + RES_DIR + SYSTEM_NAME + os.sep

#------------------------------------------------------------------------------
# network definition
#------------------------------------------------------------------------------
class SRNet(ChainList):
    def __init__(self):
        super(SRNet, self).__init__()
        for d in range(NW_DEPTH):
            self.add_link(L.Convolution2D(3 if d == 0 else NW_UNITS,
                                          (12 if SR_METHOD == "sp" else 3)
                                          if d == NW_DEPTH - 1 else NW_UNITS,
                                          NW_KSIZE, pad = (NW_KSIZE - 1) // 2))
    def forward(self, x_data, s_data):
        orig = Variable(x_data)
        x = F.leaky_relu(self[0](orig), slope = LRELU_SLOPE)
        for d in range(1, NW_DEPTH - 1):
            x = F.leaky_relu(self[d](x), slope = LRELU_SLOPE) + x
        if SR_METHOD == "sp":
            orig = F.concat([orig,] * 4, axis = 1)
        x = self[-1](x) + orig
        if chainer.config.train:
            return F.mean_squared_error(x, Variable(s_data))
        else:
            return cuda.to_cpu(x.data)

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
# loads image <index>.png from <path>
# - returns: np.ndarray(height, width, RGB/HLS) uint8
# - <height> and <width> will be adjusted to an even number
#------------------------------------------------------------------------------
def load_img(index, dlen, path, hls = False):
    # generates file name
    if dlen != 0:
        fname = "{0:s}{1:>0{dlen}d}.png".format(path, index, dlen = dlen)
    else:
        fname = "{0:s}{1:d}.png".format(path, index)
    # loads and adjust image size
    img = cv2.imread(fname)
    msg = "load_img: [{0:s}] ".format(fname)
    if len(img.shape) != 3:
        error_exit(msg + "unexpected image format")
    if (np.amin(img) < 0) or (255 < np.amax(img)):
        error_exit(msg + "unexpected value detected")
    if (img.shape[0] < INPUT_SIZE * 2) or (img.shape[1] < INPUT_SIZE * 2):
        error_exit(msg + "image size is too small")
    img = img[:(img.shape[0] // 2) * 2, :(img.shape[1] // 2) * 2]
    if hls:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)

#------------------------------------------------------------------------------
# prepares an input image (scaling down and reshaping)
# - input  : np.ndarray(height, width, 3)
# - returns: np.ndarray(1, 3, height, width)
#------------------------------------------------------------------------------
def prepare_input_img(in_img, comp = False, lower = 20, upper = 80):
    img = cv2.resize(in_img, (in_img.shape[1] // 2, in_img.shape[0] // 2),
                     interpolation = cv2.INTER_AREA)
    if comp:
        res, img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                              random.randint(lower, upper)])
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if SR_METHOD == "pe":
        img = cv2.resize(img, (in_img.shape[1], in_img.shape[0]),
                         interpolation = cv2.INTER_CUBIC)
    return img.transpose(2, 0, 1)[np.newaxis, :, :, :]

#------------------------------------------------------------------------------
# prepares an input image for evaluation
# - returns: np.ndarray(1, 3, height, width)  (sp, pe)
#            np.ndarray(4, 3, height, width)  (fp)
#------------------------------------------------------------------------------
def prepare_eval_img(in_img):
    img = prepare_input_img(in_img, JPEG_COMP, 50, 50)
    if SR_METHOD == "fp":
        img = np.concatenate([img, img[:, :, :, ::-1], img[:, :, ::-1, :],
                              img[:, :, ::-1, ::-1]], axis = 0)
    return img

#------------------------------------------------------------------------------
# generates [network input image] and [supervisory image]
#------------------------------------------------------------------------------
def generate_train_data(in_img):
    # random cropping
    trim_size = INPUT_SIZE if SR_METHOD == "pe" else INPUT_SIZE * 2
    if COLOR_SHIFT:
        trim_size += 2
    v  = random.randint(0, in_img.shape[0] - trim_size)
    h  = random.randint(0, in_img.shape[1] - trim_size)
    sv = in_img[v:v+trim_size, h:h+trim_size].copy()
    # random flip/rotation augmentation
    if random.randint(0, 1) == 1:
        sv = sv[:, ::-1]
    if random.randint(0, 1) == 1:
        sv = sv[::-1, :]
    if random.randint(0, 1) == 1:
        sv = np.rot90(sv)
    # HLS color shift augmentation
    if COLOR_SHIFT:
        sv = sv.astype(np.float32)
        sv[:, :, 0] = (sv[:, :, 0] + np.random.randn() * 32.) % 180.
        sv[:, :, 1] =  sv[:, :, 1] + np.random.randn() * 32.
        sv[:, :, 2] =  sv[:, :, 2] + sv[:, :, 2] * np.random.randn() * 0.25
        sv = np.clip(sv, 0., 255.).astype(np.uint8)
        sv = cv2.cvtColor(sv, cv2.COLOR_HLS2RGB)[1:-1, 1:-1]
    # creates input and supervisory images (JPEG compression augmentation)
    tr = uint8_to_float(prepare_input_img(sv, JPEG_COMP))
    sv = uint8_to_float(sv.transpose(2, 0, 1))[np.newaxis, :, :, :]
    if SR_METHOD == "sp":
        sv = F.space2depth(sv, 2).data
    elif SR_METHOD == "fp":
        sv = sv[:, :, ::2, ::2]
    if DEBUG:
        show_img(float_to_uint8(tr[0, 0:3].transpose(1, 2, 0)), "train_batch")
        show_img(float_to_uint8(sv[0, 0:3].transpose(1, 2, 0)),
                 "sv_batch", 1.5)
    # reshaping and type conversion
    return tr, sv

#------------------------------------------------------------------------------
# feeds training data
#------------------------------------------------------------------------------
def feed_data():
    # loads images and prepares dataset
    res_q.put("disp", "loading training images")
    res_q.put("prog_set", TRAIN_NUM)
    tr_img = pool_map(load_img, TRAIN_NUM, True, DIGIT_LEN, TRAIN_DIR,
                      COLOR_SHIFT)
    # prepares an iterator
    serial_iter = SerialIterator(tr_img, BATCH_SIZE, shuffle = True)
    data_q.put("prepare")
    for b in range(RESTART_POS, MAX_BATCH):
        batch = serial_iter.next()
        batch = [generate_train_data(x) for x in batch]
        tb = np.concatenate([x[0] for x in batch], axis = 0)
        sb = np.concatenate([x[1] for x in batch], axis = 0)
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
        model = SRNet()
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
            # prepares evaluation dataset
            res_q.put("disp", "preparing evaluation dataset")
            res_q.put("prog_set", EVAL_NUM * 2)
            ev_sv = pool_map(load_img, EVAL_NUM, True, DIGIT_LEN, EVAL_DIR)
            ev_in = pool_map(prepare_eval_img, EVAL_NUM, False, ev_sv)
        # evaluation
        elif msg == "eval":
            eval_q.put("eval_start", data)
            res_q.put("disp", "evaluating @ batch {:d}".format(data))
            res_q.put("prog_set", EVAL_NUM * 2)
            # performs forward propagation and checks value distribution
            for i, nw in enumerate(ev_in):
                lr = uint8_to_float(nw)
                if 0 <= GPU_ID:
                    lr = cuda.to_gpu(lr)
                with chainer.using_config("train", False):
                    with chainer.no_backprop_mode():
                        sr = model(lr, None)
                        if   SR_METHOD == "sp":
                            sr = F.depth2space(sr, 2).data
                        elif SR_METHOD == "fp":
                            sr[1] = sr[1, :, ::  , ::-1]
                            sr[2] = sr[2, :, ::-1, ::  ]
                            sr[3] = sr[3, :, ::-1, ::-1]
                            sr = sr.reshape((1, 12, sr.shape[2], sr.shape[3]))
                            sr = F.depth2space(sr, 2).data
                        sr = float_to_uint8(sr[0]).transpose(1, 2, 0)
                        eval_q.put("eval_res", (sr, ev_sv[i]))
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
    max_count, max_psnr = 0, 0.0
    while True:
        msg, data = eval_q.get()
        if msg == "end":
            res_q.put("end")
            break
        elif msg == "eval_start":
            train_count = data
            results, psnrs, ssims = [], [], []
        elif msg == "eval_res":
            res, sv = data
            psnrs.append(sk_ms.compare_psnr(res, sv))
            ssims.append(sk_ms.compare_ssim(res, sv, multichannel = True))
            results.append(res)
            res_q.put("prog")
        elif msg == "eval_finish":
            total_psnr, total_ssim = np.mean(psnrs), np.mean(ssims)
            psnrs = [total_psnr] + psnrs
            ssims = [total_ssim] + ssims
            res_q.put("save_qual", (train_count, psnrs, ssims))
            if ((total_psnr > max_psnr) or
                (train_count % (FORCE_SAVE * EVAL_ITVL) == 0)):
                res_q.put("save_model", (train_count, data))
                for i, x in enumerate(results):
                    res_q.put("save_img", 
                              ("{0:08d}_{1:04d}.png".format(train_count,i), x))
            if total_psnr > max_psnr:
                max_psnr, max_count = total_psnr, train_count
            res_q.put("disp", "psnr: {0:f},".format(total_psnr) +
                      " max_psnr: {0:f}".format(max_psnr) +
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
    hst_restart(RES_DIR + "qual_psnr.txt",  RESTART_POS)
    hst_restart(RES_DIR + "qual_ssim.txt", RESTART_POS)
    while True:
        msg, data = res_q.get()
        if   msg == "disp":
            print("{0:s}\r- {1:s}".format(" " * 75, data))
        elif msg == "prog_set":
            finished, total = 0, data
        elif msg == "prog":
            finished += 1
            prog_bar(finished, total)
        elif msg == "save_qual":
            train_count, psnrs, ssims = data
            hst_update(RES_DIR + "qual_psnr.txt", train_count, psnrs)
            hst_update(RES_DIR + "qual_ssim.txt", train_count, ssims)
        elif msg == "save_model":
            train_count, model_data = data
            with open(RES_DIR +
                      "p_{0:08d}.pickle".format(train_count), 'wb') as fp:
                pickle.dump(model_data, fp)
        elif msg == "save_img":
            path, img = data
            cv2.imwrite(RES_DIR + path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
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
    if not os.path.exists(RES_DIR):
        try:
            print("- creating output directory:")
            os.makedirs(RES_DIR)
        except IOError:
            error_exit("Failed to make the directory")
    if os.path.exists(RES_DIR + "qual_ssim.txt"):
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
