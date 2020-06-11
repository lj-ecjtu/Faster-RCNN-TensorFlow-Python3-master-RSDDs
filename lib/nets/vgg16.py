# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim

import lib.config.config as cfg
from lib.nets.network import Network

class vgg16(Network):
    def __init__(self, batch_size=1):
        #在使用构造函数对实例对象初始化时，可以在子类的构造函数中调用超类的构造函数
        Network.__init__(self, batch_size=batch_size)
    
    # 此处的参数is_training为bool型
    def build_network(self, sess, is_training=True):
        # tf.variable_scope函数除了可以控制tf.get_variable执行获取/创建变量的功能，这个函数也提供了一个管理变量命名空间的方式

        # tf.name_scope函数也可以管理命名空间，唯一的区别在与使用tf.get_variabvle函数时
        # 使用tf.variable函数生成变量时，变量名称会受tf.name_scope函数的影响
        # 使用tf.get_variable函数生成变量时，变量名称不受tf.name_scope函数的影响

        with tf.variable_scope('vgg_16', 'vgg_16'):

            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Build head（VGG16）
            net = self.build_head(is_training)

            # Build rpn
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)

            # Build proposals
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            # Build predictions
            cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, is_training, initializer, initializer_bbox)

            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape   #rpn_cls_score经过reshape和softmax之后得到rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob          #rpn_cls_score_reshape再经过一次改变形状得到 rpn_cls_prob 
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob      #cls_score经过softmax之后得到cls_prob
            self._predictions["bbox_pred"] = bbox_pred   
            self._predictions["rois"] = rois

            self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == 'vgg_16/conv1/conv1_1/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)

                #node:0表示名称为node节点的第一个输出，它也可以被记为node
                #当为:0时，可以忽略
                #tensorflow可以通过“字典”将模型保存时的“变量名”和需要加载的“变量”联系起来
                restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                              "vgg_16/fc7/weights": fc7_conv,
                                              "vgg_16/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                              self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                              self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))  #第三个维度进行翻转
    
    # 此处的is_training为bool型
    def build_head(self, is_training):

        #注：这里应用slim.conv2d还没有说明weight与biase如何初始化
        # Main network
        # Layer  1
        #这里的self._image为placehold
        net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')

        # Append network to summaries
        self._act_summaries.append(net)

        # Append network as head layer
        self._layers['head'] = net

        return net

    def build_rpn(self, net, is_training, initializer):

        # Build anchor component
        self._anchor_component()

        # Create RPN Layer   #这里好像没有加池化
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")

        self._act_summaries.append(rpn)
        # 默认stride=1, padding='SAME', data_format=None, rate=1, activation_fn=nn.relu
        # 这里得出的rpn_cls_score为self._num_anchors * 2个通道，对于每一个点，先是9个前景得分，再是9个背景得分
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')

        # Change it so that the score has 2 as its channel size
        # [1,none,none,2]  第一个none应该为（行*9）,先显示所有点产生的第一种anchor，再显示所有点产生的第二种anchor
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
       
        # softmax之后又改变形状为[1,none,none,2] ，softmax是对每一行进行softmax计算
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        
        # [1,none,none,18]  18个通道中前9个为前景得分，后9个为背景得分
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        # 这里得出的rpn_bbox_pred为self._num_anchors * 4个通道，先是每一个anchor的4个坐标预测，再是下一个anchor的4个坐标预测
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):

        if is_training:
            # 返回的rois中多加了一列
            # 此处为非极大值抑制产生2000个proposal
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")

            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                #  rois (0, x1, y1, x2, y2)，coming from RPN 然后再减少至256个
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                # 根据RPN的分类得分，选出得分最高的前300个anchor，并根据RPN的rpn_bbox_pred进去第一次位置修正
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        return rois

    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):

        # Crop image ROIs
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        # slim.flatten()将输入扁平化但保留batch_size
        # 每一行是一个proposal
        pool5_flat = slim.flatten(pool5, scope='flatten')

        # Fully connected layers
        fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
        if is_training:
            fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')

        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        if is_training:
            fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')

        # Scores and predictions
        cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')

        return cls_score, cls_prob, bbox_prediction
