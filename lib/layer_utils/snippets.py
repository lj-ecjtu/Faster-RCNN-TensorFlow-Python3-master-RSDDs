# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.layer_utils.generate_anchors import generate_anchors

# 这里的height和weight为RPN的输入尺寸，为原始图片缩小16倍以后
def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]  #9
    shift_x = np.arange(0, width) * feat_stride    #feature map上的点对应回原输入图片的坐标
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # np.reval()和np.flatten()两者的功能是一致的，将多维数组降为一维
    # 但两者的区别是no.flatten()返回的是一份拷贝，对拷贝做的修改不影响原始数组，而np.reval()返回的是视图，修改时会影响原数组
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]   
    # width changes faster, so here it is H, W, C

    #anchors坐标加上加上各个中心点的坐标，即得到各个中心点产生的所有anchors的坐标
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))  
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)   #一个点产生的9个anchor连在一起
    length = np.int32(anchors.shape[0])

    return anchors, length
