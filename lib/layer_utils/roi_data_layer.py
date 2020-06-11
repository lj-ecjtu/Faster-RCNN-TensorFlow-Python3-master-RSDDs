# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from lib.config import config as cfg
from lib.utils.minibatch import get_minibatch


class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes, random=False):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        # Also set a random flag
        self._random = random
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # If the random flag is set,
        # then the database is shuffled according to system time
        # Useful for the validation set
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)
        #把原来image_index随机打乱
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        # Restore the random state
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + cfg.FLAGS.ims_per_batch >= len(self._roidb):
            self._shuffle_roidb_inds()
        
        #当前minibatch对应的打乱之后的图片索引
        db_inds = self._perm[self._cur:self._cur + cfg.FLAGS.ims_per_batch]
        self._cur += cfg.FLAGS.ims_per_batch

        return db_inds
    

    # 返回一个字典blobs，键值为：
    # data        输入的image，四维数组，第一维表示每一个minibatch中的第几张图片
    # gt_boxes    将gt_box坐标对应到图片缩放之后的坐标，前四列是坐标，第五列是类别
    # im_info     image等比缩放之后的尺寸和缩放比例
    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs
        '''
         返回一个字典blobs，键值为：
         gt_boxes    将gt_box坐标对应到图片缩放之后的坐标，前四列是坐标，第五列是类别
         im_info     image等比缩放之后的尺寸和缩放比例
        '''
