# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

from nets.vgg16 import vgg16

import torch


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--model', dest='model', help='model to test', default=None, type=str)
    parser.add_argument(
        '--imdb',
        dest='imdb_name',
        help='dataset to test',
        default='icdar_test',
        type=str)
    parser.add_argument(
        '--comp',
        dest='comp_mode',
        help='competition mode',
        action='store_true')
    parser.add_argument(
        '--num_dets',
        dest='max_per_image',
        help='max number of detections per image',
        default=100,
        type=int)
    parser.add_argument(
        '--tag', dest='tag', help='tag of the model', default='', type=str)
    parser.add_argument(
        '--net',
        dest='net',
        help='vgg16',
        default='vgg16',
        type=str)
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    # if has model, get the name from it
    # if does not, then just use the initialization weights
    if args.model:
        filename = os.path.splitext(os.path.basename(args.model))[0]
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]

    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    # load network
    if args.net == 'vgg16':
        net = vgg16()
    else:
        raise NotImplementedError

    # load model
    net.create_architecture(
        imdb.num_classes,
        tag='default',
        anchor_scales=cfg.ANCHOR_SCALES,
        anchor_ratios=cfg.ANCHOR_RATIOS)

    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)

    if args.model:
        print(('Loading model check point from {:s}').format(args.model))
        net.load_state_dict(
            torch.load(args.model, map_location=lambda storage, loc: storage))
        print('Loaded.')
    else:
        print(('Loading initial weights from {:s}').format(args.weight))
        print('Loaded.')

    test_net(net, imdb, filename, max_per_image=args.max_per_image)
