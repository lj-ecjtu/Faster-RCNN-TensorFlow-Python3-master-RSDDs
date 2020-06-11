# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import numpy.random as npr

from lib.config import config as cfg
from lib.utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)   #num_images=1
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.FLAGS2["scales"]),   #数组元素都为0，大小为num_images=1
                                    size=num_images)
    #'batch_size', 256, "Network batch size during training"
    assert (cfg.FLAGS.batch_size % num_images == 0), 'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images, cfg.FLAGS.batch_size)

    # Get the input image blob, formatted for caffe
    # 以指定的比例对image进行缩放
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.FLAGS.use_all_gt:
        # Include all ground truth boxes
        # np.where(condition)返回满足条件元素的坐标，这里的坐标以tuple的形式给出，
        # 通常原数组有多少维，输出的tuple就包含几个数组，分别对应符合条件元素的各维度坐标
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)

    #将gt_box坐标对应到图片缩放之后的坐标
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]     #将gt_box坐标对应到图片缩放之后的坐标
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]                    #前四列是坐标，第五列是类别

    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    

    # blobs是一个字典，键值为：
    # data        输入的image，四维数组，第一维表示每一个minibatch中的第几张图片
    # gt_boxes    将gt_box坐标对应到图片缩放之后的坐标，前四列是坐标，第五列是类别
    # im_info     image等比缩放之后的尺寸和缩放比例
    return blobs

# 以指定的比例对image进行缩放
def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])

        #如果roidb是水平翻转过的，则读取的image也相应的水平翻转
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]    #[y，x,深度]

        target_size = cfg.FLAGS2["scales"][scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.FLAGS2["pixel_means"], target_size, cfg.FLAGS.max_size)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)    #转换list of images为numpy输入形式，四个维度

    # blob是一个四维数组，第一维表示每一个minibatch中的第几张图片
    # im_scales是一个列表，列表元素为minibatch中每一张图片的缩放比例
    return blob, im_scales
