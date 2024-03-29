import cv2
import torch
import torchvision.transforms as transforms
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
from mmdet.apis import async_inference_detector, inference_detector
from mmdet.apis.inference import init_detector

import os
from ultralytics import YOLO
import json
import random

from ensemble_boxes import *


def get_image_Id(img_name):
  img_name = img_name.split('.png')[0]
  sceneList = ['M', 'A', 'E', 'N']
  cameraIndx = int(img_name.split('_')[0].split('camera')[1])
  sceneIndx = sceneList.index(img_name.split('_')[1])
  frameIndx = int(img_name.split('_')[2])
  imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
  return imageId
def transform_horizontal_flip(boxes, img_width):
    transformed = []
    for box in boxes:
        x1, y1, x2, y2 = box
        new_x1 = img_width - x2
        new_x2 = img_width - x1
        transformed.append([new_x1, y1, new_x2, y2])
    return transformed

def transform_vertical_flip(boxes, img_height):
    transformed = []
    for box in boxes:
        x1, y1, x2, y2 = box
        new_y1 = img_height - y2
        new_y2 = img_height - y1
        transformed.append([x1, new_y1, x2, new_y2])
    return transformed
def mmdet3x_convert_to_bboxes_mmdet(results, threshold):
    boxes_list = []
    scores_list = []
    labels_list = []
    confidence_score = results.pred_instances.scores.tolist()
    for i, conf in enumerate(confidence_score):
        if conf >= threshold:
            # print(conf)
            extracted_box = results.pred_instances.bboxes[i].cpu().tolist()
            extracted_label = results.pred_instances.labels[i].cpu().tolist()
            boxes_list.append([int(extracted_box[0]),
                               int(extracted_box[1]),
                                int(extracted_box[2]),
                                int(extracted_box[3])])
            scores_list.append(conf)
            labels_list.append('{}'.format(extracted_label))
    return boxes_list, labels_list, scores_list


