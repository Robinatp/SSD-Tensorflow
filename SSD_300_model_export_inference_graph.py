import os
import math
import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim import queues

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

import sys
sys.path.append('../')

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import pascalvoc_2007
from datasets import pascalvoc_2012
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


from nets import ssd_vgg_300
from nets import ssd_vgg_512
from nets import ssd_common
import tf_extended as tfe
from preprocessing import tf_image
from preprocessing import ssd_vgg_preprocessing


def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    
    
def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[int(classes[i])]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
        
colors = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph, ["detection_scores","detection_classes","detection_boxes","num_detections"])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return

def bboxes_select(classes, scores, bboxes, threshold=0.1):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    mask = scores > threshold
    classes = classes[mask]
    scores = scores[mask]
    bboxes = bboxes[mask]
    return classes, scores, bboxes

def _bboxes_select(classes, scores, bboxes, threshold=0.1):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    mask = scores > threshold
    classes = classes[mask]
    scores = scores[mask]
    bboxes = bboxes[mask]
    return classes, scores, bboxes

def bboxes_select_mask(classes, scores, bboxes, threshold=0.1):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    mask = tf.greater(scores, threshold)
    classes = tf.boolean_mask(classes,mask)
    bboxes = tf.boolean_mask(bboxes,mask)
    scores = tf.boolean_mask(scores,mask)
    return classes, scores, bboxes


def test():
    
    ckpt_filename = 'train_logs/model.ckpt-6604'
    DATASET_DIR = 'tfrecords/'
    SPLIT_NAME = 'test'
    BATCH_SIZE = 16
    
    
    graph = tf.Graph()
    # Dataset provider loading data from the dataset.
    dataset = pascalvoc_2007.get_split(SPLIT_NAME, DATASET_DIR)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, 
                                                              shuffle=True,
    #                                                           num_epochs=1,
                                                              common_queue_capacity=2 * BATCH_SIZE,
                                                              common_queue_min=BATCH_SIZE)
    [image, shape, bboxes, labels] = provider.get(['image', 'shape', 'object/bbox', 'object/label'])
    print('Dataset:', dataset.data_sources, '|', dataset.num_samples)
    for item in provider._items_to_tensors:
        print(item, provider._items_to_tensors[item])
        
    input_shape = (None, None, None, 3)
    input_tensor = tf.placeholder(dtype=tf.uint8, shape=input_shape, name='image_tensor')   
        
    # SSD object.
    reuse = True if 'ssd' in locals() else None
    params = ssd_vgg_300.SSDNet.default_params
    ssd = ssd_vgg_300.SSDNet(params)
    
    # Image pre-processimg
    out_shape = ssd.params.img_shape
#     image_pre, labels_pre, bboxes_pre, bbox_img = \
#         ssd_vgg_preprocessing.preprocess_for_eval(image, labels, bboxes, out_shape, 
#                                                   resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_pre= tf_image.resize_image_tensor(input_tensor, out_shape,method=tf.image.ResizeMethod.BILINEAR)
    print(image_pre)
#     image_4d = tf.expand_dims(image_pre, 0)
    
    # SSD construction.
    with slim.arg_scope(ssd.arg_scope(weight_decay=0.0005)):
        predictions, localisations, logits, end_points = ssd.net(image_pre, is_training=False, reuse=reuse)
    for k in sorted(end_points.keys()):
        print(k, end_points[k].get_shape())

    # SSD default anchor boxes.
    img_shape = out_shape
    layers_anchors = ssd.anchors(img_shape, dtype=np.float32)


    # Targets encoding.Encode groundtruth labels and bboxes.gclasses, glocalisations, gscores
#     target_labels, target_localizations, target_scores = \
#                 ssd.bboxes_encode(labels, bboxes_pre, layers_anchors)
#     target_labels, target_localizations, target_scores = \
#         ssd_common.tf_ssd_bboxes_encode(labels, bboxes_pre, layers_anchors, 
#                                         num_classes=params.num_classes, no_annotation_label=params.no_annotation_label)


    # Output decoding.Detected objects from SSD output.
    localisations = ssd.bboxes_decode(localisations, layers_anchors)
    tclasses, tscores, tbboxes = \
                ssd.detected_bboxes(predictions, localisations,
                                        select_threshold=0.1,
                                        nms_threshold=0.5,
                                        clipping_bbox=None,
                                        top_k=200,
                                        keep_top_k=10)
                
    tclasses, tscores, tbboxes = bboxes_select_mask(tclasses, tscores, tbboxes, 0.1)          
    print("print1:",tclasses, tscores, tbboxes)
    
    num_detections = tf.constant([10], tf.float32)
    outputs = {}
    outputs["detection_classes"] = tf.identity(tclasses, name="detection_classes")
    outputs["detection_scores"] = tf.identity(tscores, name="detection_scores")
    outputs["detection_boxes"] = tf.identity(tbboxes, name="detection_boxes") 
    outputs["num_detections"] = tf.identity(num_detections, name="num_detections")           
                
    
    
    #medthod 2 nms batch all classes
    # Select top_k bboxes from predictions, and clip          
    fclasses, fscores, fbboxes = \
                 ssd_common.tf_ssd_bboxes_select_all_classes(predictions, localisations,
                                     select_threshold=0.1)
    #sort by scores
    rclasses_sort,rscores_sort, rbboxes_sort = \
            tfe.bboxes_sort_all_classes(fclasses,fscores, fbboxes, top_k=200)
    # Apply NMS algorithm.
    rclasses_sort,rscores_sort, rbboxes_sort = \
            tfe.bboxes_nms_batch_all_classes(rclasses_sort,rscores_sort, rbboxes_sort,
                                 nms_threshold=0.5,
                                 keep_top_k=20)
    print("print2:",rclasses_sort, rscores_sort, rbboxes_sort)
    
    
    
    with tf.Session(graph = tf.get_default_graph()) as session:
#         session.run(tf.global_variables_initializer())
        
        # Restore SSD model.
        saver = tf.train.Saver()
        saver.restore(session, ckpt_filename)
        
        save_graph_to_file(session,session.graph_def ,"new_freeze_graph.pb")
#         
#         
#         # with queues.QueueRunners(sess):
#         # Start populating queues.
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         
#         for l in range(provider._num_samples):
#             # Run model.
#             [rimg, rpredictions, rlocalisations,  rclasses, rscores, rbboxes, \
#              glabels, gbboxes, rbbox_img, \
#              rfclasses, rfscores, rfbboxes,\
#              rt_labels, rt_localizations, rt_scores] = \
#                 session.run([image_4d, predictions, localisations, tclasses, tscores, tbboxes,
#                            labels, bboxes_pre, bbox_img, 
#                            rclasses_sort, rscores_sort, rbboxes_sort,
#                            target_labels, target_localizations, target_scores])
#             
#             rfclasses,rfscores, rfbboxes = bboxes_select(rfclasses,rfscores, rfbboxes,0.5)#(rscores, rbboxes, 0.1)
#             print(rfclasses,rfscores, rfbboxes)
#             
#             
#             rclasses, rscores, rbboxes = _bboxes_select(rclasses, rscores, rbboxes,0.5)#(rscores, rbboxes, 0.1)
#             print(rclasses, rscores, rbboxes)
#             
#         
#         
# #         fig = plt.figure(figsize = (10,10))
# #         plt.imshow(np.squeeze(img_bboxes))
# #         plt.show()
#         
#             # Draw bboxes
#             img_bboxes = np.copy(ssd_vgg_preprocessing.np_image_unwhitened(rimg[0]))
# #             bboxes_draw_on_img(img_bboxes, rfclasses, rfscores, rfbboxes, colors_tableau, thickness=1)
#             bboxes_draw_on_img(img_bboxes, rclasses, rscores, rbboxes, colors_tableau, thickness=1)
# #             bboxes_draw_on_img(img_bboxes, glabels, np.zeros_like(glabels), gbboxes, colors_tableau, thickness=1)
#             # bboxes_draw_on_img(img_bboxes, test_labels, test_scores, test_bboxes, colors_tableau, thickness=1)
#          
# #         print('Labels / scores:', list(zip(rclasses, rscores)))
# #         print('Grountruth labels:', list(glabels))
# #         print(gbboxes)
# #         print(rbboxes)
#             fig = plt.figure(figsize = (10,10))
#             plt.imshow(img_bboxes)
#             plt.show()
#         
# #         for l in range(provider._num_samples):
# #             # Draw groundtruth bounding boxes using TF routine.
# #             image_bboxes = tf.squeeze(tf.image.draw_bounding_boxes(tf.expand_dims(tf.to_float(image) / 255., 0), 
# #                                                                    tf.expand_dims(bboxes, 0)))
# #             
# #             
# #             # Eval and display the image + bboxes.
# #             rimg, rshape, rbboxes, rlabels = session.run([image_bboxes, shape, bboxes, labels])
# #             
# #             print('Image shape:', rimg.shape, rshape)
# #             print('Bounding boxes:', rbboxes)
# #             print('Labels:', rlabels)
# #             
# #             fig = plt.figure(figsize = (10,10))
# #             plt.imshow(rimg)
# #             plt.show()
#         
#         coord.request_stop()
#         coord.join(threads)
    
    
test()