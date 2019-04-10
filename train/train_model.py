import argparse,logging,os,sys
import random
import mxnet as mx
from  mxnet.io import DataBatch,DataIter
from common import fit
import numpy as np
import gzip,struct
from importlib import import_module
from symbol_renasnet import get_symbol as symbol

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_iterator(args,kv):
    image_shape=(3,224,224)
    rgb_mean=np.array([123.68,116.78,103.94])
    std_scale=0.017
    data_nthreads=12

    gpu_devs = None
    if args.use_gpu_augmenter:
        gpu_devs = None if args.gpus is None or args.gpus is '' else [
                mx.gpu(int(i)) for i in args.gpus.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "train_256_q95.rec") if args.aug_level == 1
                          else os.path.join(args.data_dir, "train_480_q95.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        pad                 = 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        preprocess_threads  = data_nthreads,
        max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale    = 1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0 if args.aug_level == 1 else 0.25,
        max_rotate_angle    = 0, #if args.aug_level <= 2 else 10,
        max_shear_ratio     = 0, #if args.aug_level <= 2 else 0.1,
        rand_mirror         = True,
        shuffle             = True,
        shuffle_chunk_size  = 128,
        mean_r		        = rgb_mean[0],
        mean_g		        = rgb_mean[1],
        mean_b              = rgb_mean[2],
        scale               = std_scale,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "val_256_q95.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        rand_crop           = False,
        rand_mirror         = False,
        round_batch         = False,
        mean_r	            = rgb_mean[0],
        mean_g		        = rgb_mean[1],
        mean_b              = rgb_mean[2],
        scale               = std_scale,
        num_parts           = 1,
        part_index          = 0)
    return(train, val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training quantization model")
    parser.add_argument('--data-dir', type=str, default='hdfs://hobot-bigdata/user/qian01.zhang/mxnet_data', help='the input data directory')
    parser.add_argument('--list-dir', type=str, default='hdfs://hobot-bigdata/user/qian01.zhang/mxnet_data',
                        help='the directory which contain the training list file')
    parser.add_argument('--workspace', type=int, default=512, help='memory space size(MB) used in convolution, if xpu '
                        ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--num-classes', type=int, default=1000, help='the class number of your task')
    parser.add_argument('--aug-level', type=int, default=3, choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--num-examples', type=int, default=1281167, help='the number of training examples')
    parser.add_argument('--use-aux', default=True, action ='store_true', help='use_aux_head')
    parser.add_argument('--use-python-iter', type=int, default=0,
                      help='whether or not use python version iter')
    parser.add_argument('--use-gpu-augmenter', type=int, default=0,
                      help='whether or not augment the image in gpus')
    parser.add_argument('--color-aug', type=int, default=0,
                        help='whether or not augment the color')
    parser.add_argument('--lighting-aug', type=int, default=0,
                        help='whether or not augment the lighting')
    fit.add_fit_args(parser)
    args = parser.parse_args()
    logging.info(args)

    def get_symbol(is_training=True):

        return symbol(model_type='renasnet-ImageNet',
                          classes=args.num_classes,
                          use_aux_head = args.use_aux,
                          is_training=is_training)
    fit.fit(args,get_symbol,get_iterator)
