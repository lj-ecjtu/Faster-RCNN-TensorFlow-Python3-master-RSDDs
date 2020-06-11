#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
import lib.config.config as cf 

CLASSES = ('__background__', 'defect')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_{:s}.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def load_image_set_index():
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
     # self._devkit_path/VOC2007/ImageSets/Main/trainval.txt
    image_set_file = os.path.join(cf.FLAGS2["root_dir"],'DateSet','RSDDs_LJ','NEUTest','VOCDevkit2007',"VOC2007",'ImageSets','Main','test.txt')
   
    assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
            # f.read()把整个文件内容读进一个字符串
            # f.readline()读一行
            # f.readlines()把整个文件让你读入一个字符串列表，每一行为一个字符串
         image_index = [x.strip() for x in f.readlines()]
    return image_index
def image_path_from_index(index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(cf.FLAGS2["root_dir"],'DateSet','NEUDet_LJ','NEUTest','VOCdevkit2007','VOC2007','JPEGImages',
                                  index + ".jpg")
    assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
    return image_path    

def demo(sess, net, image_name,output_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    image_path=image_path_from_index(image_name)
    im = cv2.imread(image_path)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    # 此处的boxes是经过bbox_pre修正过的Bbox的位置坐标，并且对于预测的每一个类别，都有一个预测的Bbox坐标
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    #对每个类别进行一次画图
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #利用非极大值抑制，从300个proposal中剔除掉与更大得分的proposal的IOU大于0.1的proposal
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]   
        dets = dets[inds,:]


        output_dir=os.path.join(output_path,"comp3_det_test_{:s}.txt".format(cls))
        with open(output_dir,'a') as f:
             for i in range(len(dets)):
                 bbox = dets[i, :4]
                 score = dets[i, -1]
                 bbox_result="%s\t%f\t%f\t%f\t%f\t%f\n"%(image_name,score,bbox[0],bbox[1],bbox[2],bbox[3])   
                 f.write(bbox_result) 


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    # choices： 参数的取值范围  default：  参数的默认值  help：参数的说明信息
    # dest： 添加对象的属性名，由parse_Args()返回
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


def test_result(sess,net,tfmodel,ite):
 
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    
    output_path=os.path.join(os.path.dirname(__file__), 'mAP',"test_result\\RSDDs_1_train_test_nms\\NEU_train_test_{:s}_0.001")
    output_path=output_path.format(ite)
    print("save result in  {:s}".format(output_path))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    image_index=load_image_set_index()
    
    for i in range(len(image_index)):
        demo(sess, net, image_index[i],output_path)    

#以双下划线“__”开头和结尾的是python的内置属性
#当作为导入模块使用时，模块__name__属性值为模块的主名，当作为顶层模块直接使用时，__name__属性值为__main__
if __name__ == '__main__':
    #模块独立执行，所以执行下面的代码
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset


    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess=tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
        # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes) 
    '''
    self._predictions["bbox_pred"] *= stds
    self._predictions["bbox_pred"] += means
    '''
    net.create_architecture(sess, "TEST", n_classes,tag='default', anchor_scales=[8, 16, 32])
    
    iters=["10000","20000","30000","40000","50000","60000","70000","80000"]
    
  
    

    for i in iters:
        '''
        加载训练好的模型的路径
        dataset为键值=pascal_voc，DATASETS[dataset]=('voc_2007_trainval',)，是一个集合，所以[0]是下标
        DATASETS[dataset][0]='voc_2007_trainval'
        '''
        tfmodel = os.path.join(cf.FLAGS2["root_dir"], 'Module', 'LJModule', 'Faster RCNN', demonet, 'RSDD_1', 'default', NETS[demonet][0])
        tfmodel=tfmodel.format(i)
        if not os.path.isfile(tfmodel + '.meta'):
             print(tfmodel)
             raise IOError(('{:s} not found.\nDid you download the proper networks from our server and place them properly?').format(tfmodel + '.meta'))
        test_result(sess,net,tfmodel,i)

        

