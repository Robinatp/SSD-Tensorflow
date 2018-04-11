#coding=utf-8
import tensorflow as tf

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from datasets import dataset_factory

slim = tf.contrib.slim

# labels_to_class =['none','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor']

labels_to_class =['none','hat', 'attention']


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

def bboxes_draw_on_img(img, classes, bboxes, colors, thickness=1):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s' % (labels_to_class[classes[i]])
        p1 = (p1[0]+15, p1[1]+5)
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)

def test():

#     dataset = dataset_factory.get_dataset(
#             "pascalvoc_2007", "test", "tfrecords/")
    dataset = dataset_factory.get_dataset(
            "my_dataset", "val", "tfrecords/")
    #provider对象根据dataset信息读取数据
    provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset,
                        num_readers=3,
                        common_queue_capacity=20 * 1,
                        common_queue_min=10 * 1,
                        shuffle=False)
    
    print(provider._num_samples)
    for item in provider._items_to_tensors:
        print(item, provider._items_to_tensors[item])
        
    [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                     'object/label',
                                                     'object/bbox'])
    print (type(image))
    print (image.shape)
    print(shape,glabels,gbboxes)
    
    colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=len(labels_to_class))
    
    init=tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print('Start verification process...')
        for l in range(provider._num_samples):
            enc_image = tf.image.encode_jpeg(image)
            img, shaps, labels, boxes = session.run([enc_image, shape, glabels, gbboxes])
            f = tf.gfile.FastGFile('out.jpg' , 'wb')
            f.write(img)
            f.close()
            for j in range(labels.shape[0]):
                print('label=%d class(%s) at bbox[%f, %f, %f, %f]' % (
                    labels[j], labels_to_class[labels[j]], 
                    boxes[j][0], boxes[j][1],boxes[j][2],  boxes[j][3]))
            
            img = cv2.imread('out.jpg' )
            
            
            bboxes_draw_on_img(img, labels, boxes, colors_plasma)
            cv2.imshow('Object Detection Image',img)
            cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)
test()
