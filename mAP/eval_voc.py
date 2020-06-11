# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import print_function
import argparse
import xml.etree.ElementTree as ET
import os,sys
import pickle
import numpy as np
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='mAP Calculation')
    # choices： 参数的取值范围  default：  参数的默认值  help：参数的说明信息
    # dest： 添加对象的属性名，由parse_Args()返回
    parser.add_argument('--path', dest='path', help='The data path', type=str,default=os.path.dirname(__file__))
    args = parser.parse_args()

    return args
    
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    # 把一张图片的所有目标以字典的形式存入列表
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    # 把一张图片的所有目标以字典的形式存入列表
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):     # range(start,end,step)
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])    #每一个元素的值都等于它和它之后的值之间的最大值

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]      #找到当前元素和下一个元素不相等时的元素索引

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt  加载标签
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'VOC2007_annots_trainval.pkl')   ##此时要注意读到的文件对不对
    # read list of images
    with open(imagesetfile, 'r') as f:
        # 把整个文件内容读入一个字符串列表，每一行为一个字符串
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots，存储每一张图片的标签
        recs = {}
        for i, imagename in enumerate(imagenames):
            # 读出的每一张图片的标签是以列表的形式返回，列表元素是这张图片中每个目标的一个字典
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            # 将所有图片的标注加载为字典形式，键值为图片的标号imagename，值为对应这张图片的一个列表
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # 找到某一张图片中这个类别的目标
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])                             # 每一个Bbox为一行，即为N*4的数组
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)

        # det用来判断对应的该gt_truth是否已经被检测出来，如果已经被检测出来，则此Bbox标记为FP
        det = [False] * len(R)                                              # 列表重复 [1,2]*2=[1,2,1,2]
        """
         "~"按位取反运算符,但对于bool型，True-Flase兑换
        c=np.array([True,False,True,False])
        d=~c   [False  True False  True]
        """
        #此处npos即为TP+FN的值，所有非 difficult 的gt_truth的个数
        npos = npos + sum(~difficult)     #当为bool型时，sum（）得出的是True的个数，对应得到的是非 difficult 的gt_truth的个数                              
        # 得到某一张图片中这个类别的目标，并以字典的形式保存
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets  读结果文件
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    """
    comp3_det_test_car.txt:
	000004 0.702732 89 112 516 466
    """
    splitlines = [x.strip().split('\t') for x in lines]    #读处的每一行原先以‘\t'分隔开，处理后保存为一个列表

    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    """
    读出的每一行的为一个列表元素，这个元素又是一个Bbox坐标的列表
    也即N*4的数组
    """
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) 

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]                             # 根据置信度的高低，对Bbox坐标进行对应的调整
    image_ids = [image_ids[x] for x in sorted_ind]     # 根据置信度的高低，对image_ids进行对于的调整

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # 对于结果文件中的每一个Bbox，去找到对于图片的这个类别的所有gt_truth，然后去判断该Bbox为TP还是FP
    for d in range(nd):
        R = class_recs[image_ids[d]]  # 根据结果文件的图片标号，找到对于这张图片的标注
        bb = BB[d, :].astype(float)   # 结果文件中的一行对于的一个Bbox
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)  # 这张图片中这个类别的所有gt_truth

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
            # 计算得出该Bbox与这个类别的每一个gt_truth的IOU，并找到最大值
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            # 如果对应的是difficult目标，则该Bbox忽略不计
            if not R['difficult'][jmax]:
                
                if not R['det'][jmax]:   # 判断对应的该gt_truth是否已经被检测出来，如果已经被检测出来，则此Bbox标记为FP
                    tp[d] = 1.
                    R['det'][jmax] = 1   # 表面该gt_truth被检测出来
                else:
                    fp[d] = 1.
                '''
                tp[d] = 1.
                '''
                
        else:
            fp[d] = 1.

    # compute precision recall
    """
    a=[1,2,3,4,5,6]
    b=np.cumsum(a)   [ 1  3  6 10 15 21]
    """
    #此处fp[-1]即为fp的个数，tp[-1]即为tp的个数
    fp = np.cumsum(fp)  #原始的tp、fp本身是数组，长度与所有Bbox的个数一致
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
    


def _do_python_eval(res_prefix, output_dir = 'output'):
    _root_dic=os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..','..','..'))   #f://DeepLearning
    _devkit_path = os.path.join(_root_dic,'DateSet','RSDDs_dataset_LJ','RSDDsTest','VOCdevkit2007','VOC2007')
    _year = '2007'
    _classes = ('__background__','defect') 
    #测试结果地址
    res_prefix = os.path.join(res_prefix, 'test_result','RSDDs_2_train_test_nms','RSDDs_train_test_20000_30000_0.1','comp3_det_test_')
    filename = res_prefix + '{:s}.txt'
    #标签地址
    annopath = _devkit_path+'\\Annotations'+'\\{:s}.xml'
    #image标号地址
    imagesetfile = os.path.join(
        _devkit_path,
        'ImageSets',
        'Main',
        'test.txt')
    #标签缓存地址
    cachedir = os.path.join(os.path.dirname(__file__),'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = True if int(_year) < 2010 else False
    use_07_metric = False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(_classes):
        if cls == '__background__':
            continue
        
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        # 将单个类别的计算结果保存到本地
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)    # pkl以字典的形式保存
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('~~~~~~~~')
    print('~~~~~~~~')

if __name__ == '__main__':
    args = parse_args()

    output_path = 'mAP\\AP_result\\RSDDs_2_train_test_nms\\RSDDs_train_test_20000_30000_0.1'
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    _do_python_eval(args.path, output_dir =output_path)
