import time

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import lib.config.config as cfg
from lib.datasets import roidb as rdl_roidb
from lib.datasets.factory import get_imdb
from lib.datasets.imdb import imdb as imdb2
from lib.layer_utils.roi_data_layer import RoIDataLayer
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

try:
  import cPickle as pickle
except ImportError:
  import pickle
import os

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if True:
        #附加水平翻转训练实例
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def combined_roidb(imdb_names):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
        #实例化一个pascal_voc类对象
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))

        #为imbd.roidb_handler赋值为self.gt_roidb
        imdb.set_proposal_method("gt")
        print('Set proposal method: {:s}'.format("gt"))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = imdb2(imdb_names, tmp.classes)
    else:
        #实例化一个pascal_voc类对象
        imdb = get_imdb(imdb_names)
    return imdb, roidb


class Train:
    def __init__(self):

        # Create network
        if cfg.FLAGS.network == 'vgg16':
            self.net = vgg16(batch_size=cfg.FLAGS.ims_per_batch)
        else:
            raise NotImplementedError
        '''
        这里应用factory.py中的get_imdb(name)函数，
        然后值为 __sets[name]()，是一个字典， __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))，值为一个函数，也即实例化一个pascal_voc(split, year)对象
        也就确定了imbd的name，这里只是跟imbd的name有关，可以不用管
        '''
        self.imdb, self.roidb = combined_roidb("voc_2007_trainval")     
        
        # 对原始image和gt_boxes进行平移缩放等处理，得到network input
        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        # 模型保存的路径
        self.output_dir = cfg.get_output_dir(self.imdb, 'default')


    def train(self):

        # Create session
        #allow_soft_placement=True自动将无法放到GPU上的操作放回到CPU
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        #让GPU按需分配，不一定占用某个GPU的全部内存
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        with sess.graph.as_default():

            tf.set_random_seed(cfg.FLAGS.rng_seed)
            layers = self.net.create_architecture(sess, "TRAIN", self.imdb.num_classes, tag='default')
            loss = layers['total_loss']
            lr = tf.Variable(cfg.FLAGS.learning_rate, trainable=False)   # 0.001
            momentum = cfg.FLAGS.momentum                                # 0.9
            optimizer = tf.train.MomentumOptimizer(lr, momentum)

            gvs = optimizer.compute_gradients(loss) 

            # Double bias
            # Double the gradient of the bias if set
            if cfg.FLAGS.double_bias:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult'):
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.FLAGS.double_bias and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = optimizer.apply_gradients(final_gvs)
            else:
                train_op = optimizer.apply_gradients(gvs)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            
            # tensorboard文件保存地址
            self.tbdir=os.path.join(os.path.dirname(__file__),'log','log_RSDDs_3_40000_30000_0.1')
            if not os.path.exists(self.tbdir):
                os.makedirs(self.tbdir)
            # Write the train and validation information to tensorboard
            writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            # valwriter = tf.summary.FileWriter(self.tbvaldir)

        # Load weights

        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))

        # 从这里加一个判断，如果ckpt文件没有，则加载VGG16参数，如果由，则加载训练好模型的参数
        tfmodel = os.path.join(cfg.FLAGS2["root_dir"], 'Module', 'LJModule', 'Faster RCNN', 'vgg16', 'RSDDs_3', 'default')
        iter_add=0
        # 得到checkpointstate类
        ckpt=tf.train.get_checkpoint_state(tfmodel)
        # ckpt.model_checkpoint_path属性保存了最新模型文件的绝对文件名
        if ckpt and ckpt.model_checkpoint_path:
             saver = tf.train.Saver()
             saver.restore(sess, ckpt.model_checkpoint_path)
             tfmodel_name=ckpt.model_checkpoint_path.split('\\')[-1]
             print('Loaded  network参数 from {:s}.'.format(tfmodel_name))
             iter_add=int(ckpt.model_checkpoint_path.split('\\')[-1].split("_")[-1].split(".")[0])
             print(iter_add)
        else:
             """
             这里是将VGG16的预训练权重全部加载，原论文中是从conv3_1开始
             """
             # Fresh train directly from ImageNet weights
             print('Loading initial model weights from {:s}'.format(cfg.FLAGS.pretrained_model))

             #从VGG_16.ckpt获得权重参数名及维度，以字典的方式返回
             var_keep_dic = self.get_variables_in_checkpoint_file(cfg.FLAGS.pretrained_model)

             # Get the variables to restore, ignorizing the variables to fix（修理)
             variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

             restorer = tf.train.Saver(variables_to_restore)
             restorer.restore(sess, cfg.FLAGS.pretrained_model)  # 加载conv1到conv5参数
             print('Loaded VGG16参数.')

             # Need to fix the variables before loading, so that the RGB weights are changed to BGR
             # For VGG16 it also changes the convolutional weights fc6 and fc7 to
             # fully connected weights
             self.net.fix_variables(sess, cfg.FLAGS.pretrained_model)
             print('Fixed.')

        sess.run(tf.assign(lr, cfg.FLAGS.learning_rate))
        last_snapshot_iter = iter_add   #如果是从训练好的模型加载参数继续训练时，这里显示迭代次数
        
        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        while iter < cfg.FLAGS.max_iters + 1:
            # Learning rate
            # FLAGS.step_size=60000：Step size for reducing the learning rate, currently only support one step
            if iter == cfg.FLAGS.step_size_1 + 1:
                # Add snapshot here before reducing the learning rate
                # self.snapshot(sess, iter)
                sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * cfg.FLAGS.gamma))  # 0.001*0.1
            '''
            if iter == cfg.FLAGS.step_size_2 + 1:
                # Add snapshot here before reducing the learning rate
                # self.snapshot(sess, iter)
                sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * cfg.FLAGS.gamma * cfg.FLAGS.gamma))  # 0.001*0.1*0.1
            '''
            timer.tic()
            # Get training data, one batch at a time
            '''
            返回一个字典blobs，键值为：
            data        输入的image，四维数组，第一维表示每一个minibatch中的第几张图片
            gt_boxes    将gt_box坐标对应到图片缩放之后的坐标，前四列是坐标，第五列是类别
            im_info     image等比缩放之后的尺寸和缩放比例
            '''
            blobs = self.data_layer.forward()

            # Compute the graph without summary
            try:
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss= self.net.train_step(sess, blobs, train_op)
                
            except Exception:
                # if some errors were encountered image is skipped without increasing iterations
                print('image invalid, skipping')
                continue

            timer.toc()
            



            # Display training information
            if iter % (cfg.FLAGS.display) == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n ' % \
                      (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            # 每迭代50次，将所有日志写入文件，tensorboard就可以拿到这次运行所对应的运行信息
            if iter % 50 == 0:
                # 每迭代10次，将所有日志写入文件，tensorboard就可以拿到这次运行所对应的运行信息
                summary=self.net.get_summary_2(sess, blobs)
                writer.add_summary(summary,iter)

            #每迭代cfg.FLAGS.snapshot_iterations次，保存一次模型。
            if iter % cfg.FLAGS.snapshot_iterations == 0:
                self.snapshot(sess, iter )

            iter += 1
            
    #从VGG_16.ckpt获得权重参数名及维度，以字典的方式返回
    #{"global_step":[],"vgg_16/conv1/conv1_1/weights":[3,3,3,64],........}
    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")
                      
    #用于保存训练好的模型
    def snapshot(self, sess, iter):
        net = self.net
        #self.output_dir表示模型保存的路径，如果不存在则创建它
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)      #到config.py中修改路径
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.data_layer._cur
        # current shuffled indeces of the database
        perm = self.data_layer._perm

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename


if __name__ == '__main__':
    train = Train()
    train.train()
