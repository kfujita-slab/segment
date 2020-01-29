#-------------------------------------------------------------------------------
# <emulator>
#  - Emulator for the super-resolution system with a CNN
#    - Usage: % emulator.py [test]
#      - with "test" arguments, uses a test image generated with
#        the fixed equation for an emulation
#  - Parameter file <param.pickle> must be placed in the same directory
#-------------------------------------------------------------------------------
# Version 5.33 (August 3, 2018)
#  - Removed PRE_PADDING parameter and the corresponding code
#  - Added parameter conversion into fixed-point format
#-------------------------------------------------------------------------------
# (C) 2015-2018 Taito Manabe, all rights reserved.
#-------------------------------------------------------------------------------
# for compatibility with Python 3
from __future__      import division, print_function
from six.moves       import input, range, reduce
# migration guide:
#   1) '/' -> '//' (for an integer division)
#   2) 'print a'   -> 'print(a)', 'print a,' -> 'print(a, end = "")'
#   3) 'raw_input' -> 'input'
#   4) 'xrange'    -> 'range'
#   5) 'map(f, a)' -> 'list(map(f, a))'
#   6) exec statement -> exec function
#   7) removes L suffix for long integer
# others
import sys, os, time, itertools, math, pickle
import numpy             as np
import skimage.io        as sk_io
import skimage.transform as sk_tr
import skimage.measure   as sk_ms
import skimage.filters   as sk_fl
import chainer.functions as F
import chainer.links     as L
from chainer         import Chain
from PIL             import Image
from multiprocessing import Process, Queue, Pool

#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
# behavior
SAVE_ITM    = False                  # saves outputs of hidden layers
NO_OVF_CHK  = False                  # skips overflow check
IMG_NUM     = 30
IMG_DIR     = "dataset/eval/"        # <000.png>, <001.png>, <002.png>, ...
RES_DIR     = "sim/other/"

# network configuration
NW_FLT_SIZE = (3, 3, 3, 3)           # filter size. each must be an odd number
NW_UNITS    = (30, 20, 30)
MODULI      = ((5, 7, 9, 11, 13, 16),
               (5, 7, 9, 11, 13, 32),
               (5, 7, 9, 11, 13, 32),
               (7, 9, 11, 13, 16))
RNS_THR     = (360448, 524288, 524288, 114688)
INT_BITW    = 5
FRAC_BITW   = 8                      # fractional part bit width
SLOPE_INV   = 4                      # 1 / (slope). should be 2^n. 0: ReLU

# following parameters are calculated automatically
NW_UNITS   += (1,)
NW_DEPTH    = len(NW_FLT_SIZE)
PAD_WIDTH   = [(x-1) // 2 for x in NW_FLT_SIZE]
FILE_DIR    = os.path.dirname(os.path.abspath(__file__)) + os.sep
RES_DIR     = FILE_DIR + RES_DIR
PATCH_SIZE  = sum(NW_FLT_SIZE) - NW_DEPTH + 1
DRANGE      = [reduce(lambda a,b:a*b, m) for m in MODULI]
DR_MAX      = RNS_THR
DR_MIN      = [DR_MAX[i] - DRANGE[i] + 1 for i in range(len(DRANGE))]

#------------------------------------------------------------------------------
# network defenition for parameter loading
#------------------------------------------------------------------------------
class sr_network(Chain):
    # initialization
    def __init__(self):
        layers = ""
        for i in range(NW_DEPTH):
            ip = "1" if i == 0 else "NW_UNITS[{0:d}]".format(i-1)
            op = "NW_UNITS[{0:d}]".format(i)
            fs = "NW_FLT_SIZE[{0:d}]".format(i)
            layers += "conv{0:d} = ".format(i+1)
            layers += "L.Convolution2D({0:s}, {1:s}, {2:s})".format(ip, op, fs)
            if i < NW_DEPTH - 1:
                layers += ", "
        eval("super(sr_network, self).__init__({0:s})".format(layers))

#------------------------------------------------------------------------------
# displays progress bar
#------------------------------------------------------------------------------
def prog_bar(finished, total, width = 60, autohide = True):
    progress  = float(finished) / float(total) if (0 < total) else 0
    finished  = int(width * progress)
    remaining = width - finished
    print("  |{0:s}{1:s}|".format("=" * finished, " " * remaining) +
          " {0:3.1f}%\r".format(progress * 100.0), end = "")
    if autohide and remaining == 0:
        print("{0:s}\r".format(" " * (width + 12)), end = "")
    sys.stdout.flush()

#------------------------------------------------------------------------------
# converts <img> from float to uint8
#------------------------------------------------------------------------------
def float_to_uint8(img_in):
    img = img_in * 256.0 + 0.5
    img[np.where(img < 0.0)]   = 0.0
    img[np.where(255.0 < img)] = 255.0
    return img.astype(np.uint8)
def uint8_to_float(img_in):
    return img_in.astype(np.float32) / 256.0

#------------------------------------------------------------------------------
# loads an image <path> and converts it to a greyscale <np.float32> image
#------------------------------------------------------------------------------
def error_exit(msg):
    print(msg)
    sys.exit()
def load_one_img(path):
    try:
        img = (sk_io.imread(path, as_grey = False)).astype(np.float32)
    except IOError:
        error_exit("load_one_img: cannot load {0:s}".format(path))
    if (img.shape[0] % 2 != 0) or (img.shape[1] % 2 != 0) :
        error_exit("load_one_img: image size must be (2n x 2m)")
    if (len(img.shape) != 3) or (img.shape[2] < 3):
        error_exit("load_one_img: unsupported image format")
    if np.max(img) <= 1.0:
        img *= 255.0
    if (np.amin(img) < 0) or (255 < np.amax(img)):
        error_exit("load_one_img: unexpected value detected")
    return img[:,:,0:3].astype(np.uint8)
    
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
    with open(FILE_DIR + "param.pickle", 'rb') as fp:
        model = pickle.load(fp)
except IOError:
    error_exit("! cannot open parameter file <param.pickle>")

print("- converting parameters from float to fixed-point:")
flts   = [eval("model.conv{0:d}.W.data".format(i+1)) for i in range(NW_DEPTH)]
biases = [eval("model.conv{0:d}.b.data".format(i+1)) for i in range(NW_DEPTH)]
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

with open(RES_DIR + "fixed_param.txt", 'w') as fp:
    fp.write("")
with open(RES_DIR + "fixed_param.txt", 'a') as fp:
    for i in xrange(len(flts)):
        fp.write(fixed_to_str(flts[i],   INT_BITW + FRAC_BITW,
                              "L{0:1d}_FLT".format(i+1)))
        fp.write(fixed_to_str(biases[i], INT_BITW + FRAC_BITW * 2,
                              "L{0:1d}_BIAS".format(i+1)))

#------------------------------------------------------------------------------
# converts fixed-point parameters into the RNS
#------------------------------------------------------------------------------
# calculates gcd(a, b) using the Euclidean algorithm
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def fixed_to_rns(data, moduli, thrs):
    for i, j in itertools.combinations(moduli, 2):
        if gcd(i, j) != 1:
            print("! to_rns: {0:d} and {1:d} are not coprime".format(i, j))
            sys.exit()
    drange = reduce(lambda a,b:a*b, moduli)
    if (np.min(data) < thrs - drange + 1) or (thrs < np.max(data)):
        print("! to_rns: overflow detected")
        print("   range: [{0:d}, {1:d}]".format(np.min(data), np.max(data)))
        print(" allowed: [{0:d}, {1:d}]".format(thrs - drange + 1, thrs))
        sys.exit()
    res = np.ndarray((len(moduli),) + data.shape, dtype = np.uint8)
    for i in range(len(moduli)):
        res[i] = ((data + drange) % drange) % moduli[i]
    return res

#------------------------------------------------------------------------------
# converts RNS parameters into str (for biases)
#------------------------------------------------------------------------------
def rns_to_str(data, moduli, msg):
    res    = ""
    for i in range(len(moduli)):
        bitw = int(math.ceil(math.log(moduli[i], 2)))
        for d in np.ravel(data[i]):
            res += "{0:0>{bw}b}".format(d, bw = bitw)
    bitw   = len(res)
    res    = "{0:d}\'b".format(bitw) + res + ";\n"
    res_bw = "   localparam integer {0:s}_BITW = {1:d};\n".format(msg, bitw)
    res    = "   localparam [0:{0:d}] {1:s} = ".format(bitw-1, msg) + res
    return res_bw + res

#------------------------------------------------------------------------------
# converts RNS parameters into classes (for filters)
# [returns] res[output id][modulus][weight value] = [pos0, pos1, pos2, ...]
#------------------------------------------------------------------------------
def rns_to_cls(data, moduli):
    res  = []
    for oi in range(data.shape[1]):
        o_res = []
        for mi in range(data.shape[0]):
            d = np.ravel(data[mi, oi])
            o_res.append([np.where(d == i)[0] for i in range(moduli[mi])])
        res.append(o_res)
    return res

#------------------------------------------------------------------------------
# converts classified RNS parameters into classified str (for filters)
# format: res[modulus][output id][weight value] = [pos0, pos1, pos2, ...]
#------------------------------------------------------------------------------
def rns_to_cls_str(data, moduli, msg):
    bitw_pos = int(math.ceil(math.log(len(np.ravel(data[0, 0])), 2)))
    bitw_num = int(math.ceil(math.log(len(np.ravel(data[0, 0])) + 1, 2)))
    max_num, res_num, res_pos = 0, "", ""
    for mi in range(data.shape[0]):
        for oi in range(data.shape[1]):
            d = np.ravel(data[mi, oi])
            for i in range(moduli[mi]):
                pos = np.where(d == i)[0]
                num = len(pos)
                max_num  = max(num, max_num)
                res_num += "{0:0>{bw}b}".format(num, bw = bitw_num)
                for p in pos:
                    res_pos += "{0:0>{bw}b}".format(p, bw = bitw_pos)
    n_num   = len(res_num)
    n_pos   = len(res_pos)
    res_nbw = ("   localparam integer {0:s}_NUM_BITW = ".format(msg) +
               "{0:d};\n".format(n_num))
    res_pbw = ("   localparam integer {0:s}_POS_BITW = ".format(msg) +
               "{0:d};\n".format(n_pos))
    res_mxn = ("   localparam integer {0:s}_MAX_NUM  = ".format(msg) +
               "{0:d};\n".format(max_num))
    res_num = ("   localparam [0:{0:d}] {1:s}_".format(n_num-1, msg) +
               "NUM = {0:d}\'b".format(n_num) + res_num + ";\n")
    res_pos = ("   localparam [0:{0:d}] {1:s}_".format(n_pos-1, msg) +
               "POS = {0:d}\'b".format(n_pos) + res_pos + ";\n")
    return res_nbw + res_pbw + res_mxn + res_num + res_pos

print("- converting parameters from fixed-point to RNS:")
if not NO_OVF_CHK:
    with open(RES_DIR + "param.txt", 'w') as fp:
        fp.write("")
    with open(RES_DIR + "param.txt", 'a') as fp:
        for i in range(len(flts)):
            flt = fixed_to_rns(flts[i],   MODULI[i], RNS_THR[i])
            bs  = fixed_to_rns(biases[i], MODULI[i], RNS_THR[i])
            fp.write(rns_to_cls_str(flt, MODULI[i], "L{0:1d}_FLT".format(i+1)))
            fp.write(rns_to_str(bs, MODULI[i], "L{0:1d}_BIAS".format(i+1)))

#------------------------------------------------------------------------------
# loads / prepares test image(s)
#------------------------------------------------------------------------------
def scale_down(img):
    result = Image.fromarray(img)
    result = result.resize((result.size[0] // 2, result.size[1] // 2),
                           resample = Image.LANCZOS)
    result = np.asarray(result)
    result.flags.writeable = True
    return result

argv      = sys.argv
test_mode = (len(argv) == 2 and argv[1] == "test") 
if test_mode:
    print("- preparing the test image:")
    image_num = 1
    lr_img = np.zeros((540, 960, 3), dtype = np.uint8)
    for v, h in itertools.product(range(540), range(960)):
        v2, vh, h2 = v * v, v * h, h * h
        lr_img[v, h, 0] = (((v2 + vh) * (v + h)) // 750) % 256
        lr_img[v, h, 1] = (((v2 + h2) * (v + h)) // 750) % 256
        lr_img[v, h, 2] = (((h2 + vh) * (v + h)) // 750) % 256
    lr_img = [lr_img]
else:
    print("- loading evaluation images:")
    image_num = IMG_NUM
    img_paths = ["{0:s}{1:>0{dlen}d}.png".format(IMG_DIR, i, dlen = 3)
                 for i in range(image_num)]
    pp = Pool(4)
    hr_img = pp.map_async(load_one_img, img_paths).get(3600)
    print("- preparing low-resolution images:")
    lr_img = pp.map_async(scale_down, hr_img).get(3600)
    pp.close()
    pp.join()

#------------------------------------------------------------------------------
# prepares network input images and enlarged cbcr-images
#------------------------------------------------------------------------------
def rgb_to_ycbcr(img):
    i = img.astype(np.int32)
    r, g, b = i[:, :, 0], i[:, :, 1], i[:, :, 2]
    res = np.ndarray(img.shape, dtype = np.int32)
    res[:, :, 0] = ( 306 * r + 601 * g + 117 * b) // 1024 
    res[:, :, 1] = (-173 * r - 339 * g + 512 * b) // 1024 + 128
    res[:, :, 2] = ( 512 * r - 429 * g -  83 * b) // 1024 + 128
    if np.amin(res) < 0 or 255 < np.amax(res):
        error_exit("rgb_to_ycbcr: overflow detected")
    return res.astype(np.uint8)
def sk_rescale(img, scale, order):
    result = sk_tr.rescale(img, scale, order = order, mode = "edge")
    return result.astype(np.float32)
def bc_enlarge(img):
    return float_to_uint8(sk_rescale(uint8_to_float(img), 2.0, 3))
def bl_enlarge(img):
    return float_to_uint8(sk_rescale(uint8_to_float(img), 2.0, 1))

# converts color space and applies bicubic interpolation
pp = Pool(4)
lr_ycbcr = pp.map_async(rgb_to_ycbcr, lr_img).get(3600)
bl_ycbcr = pp.map_async(bl_enlarge, lr_ycbcr).get(3600) # y-channel is not used
lr_img   = [x[:, :, 0] for x in lr_ycbcr]
del lr_ycbcr
if not test_mode:
    bc_img = pp.map_async(bc_enlarge, lr_img).get(3600)
pp.close()
pp.join()

# applies flips
def prepare_img(img, vflip, hflip):
    fixed_img = img.astype(np.int64) * (2 ** (FRAC_BITW - 8))
    return fixed_img[::(-1 if vflip else 1), ::(-1 if hflip else 1)]
def prepare_all_imgs(imgs, vflip, hflip):
    return [prepare_img(img, vflip, hflip) for img in imgs]
lr_flp = [prepare_all_imgs(lr_img, vflip, hflip)
          for vflip, hflip in itertools.product((False, True), repeat = 2)]

#------------------------------------------------------------------------------
# limits values of <in_img> into the range of [lower, upper] 
#------------------------------------------------------------------------------
def limiter(img, lower, upper):
    minv, maxv = np.min(img), np.max(img)
    overflow   = (NO_OVF_CHK == False) and (minv < lower or upper < maxv)
    if overflow:
        print("! limiter: overflow")
        print("     vrange: [{0:d}, {1:d}]".format(minv,  maxv))
        print("    allowed: [{0:d}, {1:d}]".format(lower, upper))
        vrange = upper - lower + 1
        img    = ((img - lower) % vrange) + lower
    return img, overflow, minv, maxv

#------------------------------------------------------------------------------
# forward propagation
#------------------------------------------------------------------------------
def forward(imgs, flts, bias, lower, upper, int_bw, frac_bw):
    # input check
    imgs, overflow, minv, maxv = limiter(imgs, lower, upper)
    if overflow:
        print("! forward: overflow detected in conversion (fixed -> RNS)")
    flt_size   = flts.shape[1]
    out_height = imgs.shape[1] - flt_size + 1
    out_width  = imgs.shape[2] - flt_size + 1
    num   = (flt_size ** 2) * imgs.shape[0]
    prods = np.ndarray((num, out_height, out_width), dtype = np.int64)
    # multiplication by filter coefficients
    for i in range(imgs.shape[0]):
        img, flt = imgs[i], flts[i]
        for v, h in itertools.product(range(flt_size), repeat = 2):
            n = (i * (flt_size) + v) * flt_size + h
            prods[n] = img[v:v+out_height, h:h+out_width] * flt[v, h]
    # sums up
    sums = prods
    while num != 1:
        newnum  = (num + 3) // 4
        newsums = np.zeros((newnum,) + sums[0].shape, dtype = np.int64)
        for i in range(newnum):
            for j in range(4):
                if i*4 + j < num:
                    newsums[i] += sums[i*4 + j]
        sums = newsums
        num  = newnum
    result = sums[0]
    # adds bias
    result, overflow, tmin, tmax = limiter(result + bias, lower, upper)
    minv, maxv = min(minv, tmin), max(maxv, tmax)
    if overflow:
        print("! forward: overflow detected in conversion (RNS -> fixed)")
    # converts from RNS to binary, applies ReLU and nearest rounding
    fixed_bw = int_bw + frac_bw
    tgt      = np.where(result < 0)
    if SLOPE_INV != 0:
        result[tgt] = (result[tgt] + SLOPE_INV // 2) // SLOPE_INV
    else:
        result[tgt] = 0
    result = (result // (2**(frac_bw-1)) + 1) // 2
    result, overflow, tmin, tmax = limiter(result, -(2 ** (fixed_bw - 1)),
                                           (2 ** (fixed_bw - 1)) - 1)
    if overflow:
        print("! forward: overflow detected in binary format")
    return result, minv, maxv

#------------------------------------------------------------------------------
# convolutional neural network
#------------------------------------------------------------------------------
result_q = Queue(maxsize = 100)

def cnn(img, flptype, imgid):
    prev_res = np.array(img)
    prev_res = prev_res.reshape((1,) + prev_res.shape)
    for i in range(NW_DEPTH):
        result   = np.ndarray((NW_UNITS[i], prev_res.shape[1],
                               prev_res.shape[2]), np.int64)
        pw       = PAD_WIDTH[i]            
        prev_res = np.pad(prev_res, ((0, 0), (pw, pw), (pw, pw)), "edge")
        for j in range(NW_UNITS[i]):
            result[j], tmin, tmax = forward(prev_res, flts[i][j],
                                            biases[i][j], DR_MIN[i], DR_MAX[i],
                                            INT_BITW,  FRAC_BITW)
            if SAVE_ITM and flptype == 0:
                nrm  = tmp.astype(np.float32).copy()
                if np.max(nrm) != np.min(nrm) :
                    nrm -= np.min(nrm)
                    nrm  = ((nrm / np.max(nrm)) * 255.0).astype(np.uint8)
                sk_io.imsave(RES_DIR + str(imgid) + "_l{:d}_".format(i+1) +
                             str(i) + ".png" , nrm)
            result_q.put( ("progress", (i, tmin, tmax)) )
        prev_res = result
    return result[0]

def cnn_all(imgs, flptype):
    for i in range(len(imgs)):
        result = cnn(imgs[i], flptype, i)
        if 9 <= FRAC_BITW:
            result = (result // (2 ** (FRAC_BITW - 9)) + 1) // 2
        result[np.where(result < 0)]   = 0
        result[np.where(255 < result)] = 255
        result_q.put( ("completed", (i, result, flptype)) )
    
#------------------------------------------------------------------------------
# invokes thread
#------------------------------------------------------------------------------
print("- applying super-resolution:")
processes = [Process(target=cnn_all, args=(lr_flp[x], x)) for x in range(4)]
for p in processes:
    p.daemon = True
    p.start()

#------------------------------------------------------------------------------
# obtains results
#------------------------------------------------------------------------------
sr_y = [np.zeros((lr_img[i].shape[0] * 2, lr_img[i].shape[1] * 2), np.uint8)
        for i in range(image_num)]
minv, maxv = [], []
for i in range(NW_DEPTH):
    minv.append(0)
    maxv.append(0)
finished   = 0
total      = NW_UNITS[0] * (NW_FLT_SIZE[0] ** 2)
for i in range(1, NW_DEPTH):
    total += NW_UNITS[i-1] * NW_UNITS[i] * (NW_FLT_SIZE[i] ** 2)
total     *= 4 * image_num
completed  = 0
while True:
    while result_q.empty():
        time.sleep(0.1)
    mtype, msg = result_q.get()
    if mtype == "progress":
        depth, tmin, tmax = msg
        if depth == 0:
            finished += NW_FLT_SIZE[0] ** 2
        else:
            finished += NW_UNITS[depth - 1] * (NW_FLT_SIZE[depth] ** 2)
        if tmin < minv[depth]: minv[depth] = tmin
        if maxv[depth] < tmax: maxv[depth] = tmax
        prog_bar(finished, total, 60)
    elif mtype == "completed":
        i, result, flptype = msg
        completed += 1
        hflp = flptype  % 2
        vflp = flptype // 2
        sr_y[i][vflp::2, hflp::2] = result[::(1-vflp*2), ::(1-hflp*2)]
    if completed == 4 * image_num : break

# displays value distribution
for i in range(NW_DEPTH):
    print("    layer_{0:d}   ".format(i+1), end = "")
    print("vrange: [{0:d}, {1:d}]".format(minv[i], maxv[i]))
    
#------------------------------------------------------------------------------
# converts from YCbCr to RGB
#------------------------------------------------------------------------------
def ycbcr_to_rgb(img):
    i = img.astype(np.int32)
    y, cb, cr = i[:, :, 0], i[:, :, 1], i[:, :, 2]
    res = np.ndarray(img.shape, dtype = np.int32)
    res[:, :, 0] = (1024 * y             + 1436 * cr - 183808) // 1024 
    res[:, :, 1] = (1024 * y -  352 * cb -  731 * cr + 138624) // 1024
    res[:, :, 2] = (1024 * y + 1815 * cb             - 232320) // 1024
    res[np.where(res < 0)]   = 0
    res[np.where(255 < res)] = 255
    return res.astype(np.uint8)
# color channel reconstruction
print("- starts conversion from YCbCr to RGB")
for i in range(len(bl_ycbcr)):
    bl_ycbcr[i][:, :, 0] = sr_y[i]
pp = Pool(4)
sr_img = pp.map_async(ycbcr_to_rgb, bl_ycbcr).get(3600)
if not test_mode:
    for i in range(len(bl_ycbcr)):
        bl_ycbcr[i][:, :, 0] = bc_img[i]
    bc_img = pp.map_async(ycbcr_to_rgb, bl_ycbcr).get(3600)
pp.close()
pp.join()

#------------------------------------------------------------------------------
# evaluation
#------------------------------------------------------------------------------
if test_mode :
    print("- preparing results:")
    res_str = ""
    for v, h in itertools.product(range(sr_img[0].shape[0]),
                                  range(sr_img[0].shape[1])):
        y, cb, cr = sr_y[0][v, h], bl_ycbcr[0][v, h, 1], bl_ycbcr[0][v, h, 2]
        r, g, b   = sr_img[0][v, h]
        res_str += "({0:4d}, {1:4d}) ".format(v, h)
        res_str += "[{0:3d}, {1:3d}, {2:3d}] ".format(y, cb, cr)
        res_str += "[{0:3d}, {1:3d}, {2:3d}]\n".format(r, g, b)
    print("- saving results")
    with open(RES_DIR + "test_result.txt", 'w') as fp:
        fp.write(res_str)
    sk_io.imsave(RES_DIR + "res_test.png", sr_img[0])
else:
    print("- starts evaluation:")
    bc_ssim_total = 0.0
    bc_psnr_total = 0.0
    sr_ssim_total = 0.0
    sr_psnr_total = 0.0
    with open(RES_DIR + "qual.txt", 'w') as fp:
        fp.write("")
    for i in range(image_num):
        sk_io.imsave(RES_DIR + "sr_img_{0:02d}.png".format(i) , sr_img[i])
        bc_ssim, sr_ssim = 0.0, 0.0
        for c in range(3):
            bc_ssim += sk_ms.compare_ssim(hr_img[i][:,:,c], bc_img[i][:,:,c])
            sr_ssim += sk_ms.compare_ssim(hr_img[i][:,:,c], sr_img[i][:,:,c])
        bc_ssim /= 3.0
        sr_ssim /= 3.0
        prog_bar(i*2 + 1, image_num * 2)
        bc_psnr = sk_ms.compare_psnr(hr_img[i], bc_img[i])
        sr_psnr = sk_ms.compare_psnr(hr_img[i], sr_img[i])
        prog_bar(i*2 + 2, image_num * 2)
        with open(RES_DIR + "qual.txt", 'a') as fp:
            fp.write("[Image {:d}]\n".format(i))
            fp.write("   bicubic: {0:f} {1:f}\n".format(bc_ssim, bc_psnr))
            fp.write("  superres: {0:f} {1:f}\n".format(sr_ssim, sr_psnr))
        bc_ssim_total += bc_ssim
        bc_psnr_total += bc_psnr
        sr_ssim_total += sr_ssim
        sr_psnr_total += sr_psnr
    bc_ssim_total /= image_num
    bc_psnr_total /= image_num
    sr_ssim_total /= image_num
    sr_psnr_total /= image_num
    with open(RES_DIR + "qual.txt", 'a') as fp:
        fp.write("[Average]\n")
        fp.write("   bicubic: {0:f} {1:f}\n".format(bc_ssim_total,
                                                    bc_psnr_total))
        fp.write("  superres: {0:f} {1:f}\n".format(sr_ssim_total,
                                                    sr_psnr_total))
