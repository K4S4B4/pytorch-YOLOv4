import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from tool.utils import *
from tool.darknet2onnx import *


def main(cfg_file, weight_file, image_path, batch_size):

    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size)
    else:
        # Transform to onnx as specified batch size
        transform_to_onnx(cfg_file, weight_file, batch_size)
        # Transform to onnx as demo
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, 1)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    detect(session, image_src)



def detect(session, image_src):
    ###############################################################
    IN_IMAGE_H = session.get_inputs()[0].shape[1]
    IN_IMAGE_W = session.get_inputs()[0].shape[2]
    ###############################################################

    #IN_IMAGE_H = session.get_inputs()[0].shape[2]
    #IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    ###############################################################
    img_in = np.expand_dims(resized, axis=0).astype(np.uint8)
    #img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #img_in = img_in.astype(np.float32)
    #img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    #img_in /= 255.0
    ###############################################################
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name
    print(input_name)
    outputs = session.get_outputs()
    print(outputs[0].name)
    print(outputs[1].name)

    outputs = session.run(None, {input_name: img_in})

    #########################################################
    start = time.time()
    for i in range(10):
        outputs = session.run(None, {input_name: img_in})
    delta_time = (time.time() - start)/10
    print("onnx model inference took")
    print(delta_time)
    #########################################################

    boxes = post_processing(img_in, 0.4, 0.6, outputs)

    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)



if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if True:
    #if len(sys.argv) == 4:
    #if len(sys.argv) == 5:
        cfg_file = 'cfg/yolov4-tiny.cfg'
        weight_file = 'weight/yolov4-tiny.weights'
        image_path = 'resource/testdata/IMG_20210208_135527.jpg'
        ###############################################
        batch_size = 1 #int(sys.argv[4])
        ###############################################
        main(cfg_file, weight_file, image_path, batch_size)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <cfgFile> <weightFile> <imageFile> <batchSize>')
