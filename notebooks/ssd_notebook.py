import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

path=os.getcwd()
path=os.path.dirname(path)

# tensorflow在训练时会默认占用所有GPU的显存
# 可以通过 per_process_gpu_memory_fraction 选项设置每个GPU在进程中使用显存的上限
# 也可以通过 allow_growth=True，让所分配的显存根据需求增长
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# SSD 300 网络需要300x300的图像输入。对于任意尺寸大小的输入，网络都会将其调整为300x300（即Resize.WARP_RESIZE）。
# 尽管调整大小可能会更改比率宽度/高度，但SSD模型在调整后的图像上表现良好。
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors. 
# SSD锚点对应于网络中编码的默认边界框。SSD净输出提供这些锚的坐标和尺寸的偏移量。
net_shape = (300, 300)
#NHWC" 時，排列順序為 [batch, height, width, channels]
#NCHW" 時，排列順序為 [batch, channels, height, width]
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = path+os.sep+'checkpoints'+os.sep+'ssd_300_vgg.ckpt'+os.sep+'ssd_300_vgg.ckpt'
#ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# =========================================================================== #
# SSD网络为了提供更准确的检测结果，需要做一些后期处理，通常步骤如下：
# 1. Select boxes above a classification threshold;
# 2. Clip boxes to the image shape;
# 3. Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# 4. If necessary, resize bounding boxes to original image shape.    
# =========================================================================== #

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
path = '../demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-5])
rclasses, rscores, rbboxes =  process_image(img)

# demo目录下倒数第五张图（小轿车、自行车、狗）的输出结果如下
# rclasses
# array([ 2,  7, 12], dtype=int64)
# rscores
# array([0.99041396, 0.96858996, 0.9560676 ], dtype=float32)
# rbboxes
# array([[0.19294187, 0.18716487, 0.824463  , 0.7296326 ],
       #[0.14599293, 0.6084448 , 0.29842302, 0.9021337 ],
       #[0.38435042, 0.17588624, 0.93412066, 0.41257903]], dtype=float32)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)