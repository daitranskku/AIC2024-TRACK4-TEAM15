import cv2
import torch
import math
import numpy as np
import os
import json
import random

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision

from mmdet.apis import async_inference_detector, inference_detector
from mmdet.apis.inference import init_detector

from ultralytics import YOLO

from tqdm import tqdm

from ensemble_boxes import *
from inference_utils import *
import natsort
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

mmdet_config1 = './pretrained_weights/codetrSwinLO365_nafnet_kfold1_2048_16_3019_071/swinL_detr_o365_coco_nafnet.py'
mmdet_checkpoint1 = './pretrained_weights/codetrSwinLO365_nafnet_kfold1_2048_16_3019_071/epoch_16 (1).pth'

mmdet_config2 = './pretrained_weights/codetrSwinLO365_nafnet_kfold1_1568_12/swinL_detr_o365_coco_nafnet.py'
mmdet_checkpoint2 = './pretrained_weights/codetrSwinLO365_nafnet_kfold1_1568_12/epoch_24.pth'

mmdet_config3 = './pretrained_weights/train_val_org/codetr_nafnet_night2day_1536_0489/swinL_detr_o365_coco_nafnet.py'
mmdet_checkpoint3 = './pretrained_weights/train_val_org/codetr_nafnet_night2day_1536_0489/epoch_6_best.pth'

mmdet_config4 = './pretrained_weights/codetrSwinLO365_nafnet_kfold3_2048_16_3019_071/swinL_detr_o365_coco_nafnet (1).py'
mmdet_checkpoint4 = './pretrained_weights/codetrSwinLO365_nafnet_kfold3_2048_16_3019_071/epoch_16 (1).pth'

mmdet_config5 = './pretrained_weights/codetrSwinLO365_nafnet_kfold4_2048_16_3019_071/swinL_detr_o365_coco_nafnet (2).py'
mmdet_checkpoint5 = './pretrained_weights/codetrSwinLO365_nafnet_kfold4_2048_16_3019_071/epoch_16.pth'

mmdet_config6 = './pretrained_weights/codetrSwinLO365_nafnet_kfold5_2048_16_3019/swinL_detr_o365_coco_nafnet.py'
mmdet_checkpoint6 = './pretrained_weights/codetrSwinLO365_nafnet_kfold5_2048_16_3019/epoch_14.pth'

mmdet_config7 = './pretrained_weights/codetrSwinLO365_nafnet_kfold2_1568_16_/swinL_detr_o365_coco_nafnet.py'
mmdet_checkpoint7 = './pretrained_weights/codetrSwinLO365_nafnet_kfold2_1568_16_/epoch_10.pth'

mmdet_config8 = './pretrained_weights/train_all/codetr_trainall_nigh2day_1536/swinL_detr_o365_coco_nafnet.py'
mmdet_checkpoint8 = './pretrained_weights/train_all/codetr_trainall_nigh2day_1536/epoch_24 (1).pth'

mmdet_config9 = './pretrained_weights/train_all/pseudo_codetr_1536/swinL_detr_o365_coco_nafnet.py'
mmdet_checkpoint9 = './pretrained_weights/train_all/pseudo_codetr_1536/epoch_32.pth'

mmdet_config10 = './pretrained_weights/train_all/pseudo_codetr_1024/swinL_detr_o365_coco_nafnet.py'
mmdet_checkpoint10 = './pretrained_weights/train_all/pseudo_codetr_1024/epoch_29.pth'

# Yolo models
# yolov8x
yolo_checkpoint1 = './pretrained_weights/train_all/yolov8_trainall/best.pt'
# Yolov9e
yolov9_checkpoint1 = './pretrained_weights/train_all/yolov9_trainall/best.pt'

# Load MMDet models
# print('Loading SAHI models...')

# # Load SAHI models
# sahi_model1 = AutoDetectionModel.from_pretrained(
#     model_type='mmdet3',
#     model_path=mmdet_checkpoint10,
#     config_path=mmdet_config10,
#     confidence_threshold=0.5,
#     image_size=None, # not supported
#     device="cuda:0", # or 'cuda:0'
# )
# sahi_model2 = AutoDetectionModel.from_pretrained(
#     model_type='mmdet3',
#     model_path=mmdet_checkpoint1,
#     config_path=mmdet_config1,
#     confidence_threshold=0.5,
#     image_size=None, # not supported
#     device="cuda:0", # or 'cuda:0'
# )
print("SAHI models loaded")
mmdet_model1 = init_detector(mmdet_config1, mmdet_checkpoint1, device='cuda:0')
mmdet_model2 = init_detector(mmdet_config2, mmdet_checkpoint2, device='cuda:0')
mmdet_model3 = init_detector(mmdet_config3, mmdet_checkpoint3, device='cuda:0')
mmdet_model4 = init_detector(mmdet_config4, mmdet_checkpoint4, device='cuda:0')
mmdet_model5 = init_detector(mmdet_config5, mmdet_checkpoint5, device='cuda:0')
mmdet_model6 = init_detector(mmdet_config6, mmdet_checkpoint6, device='cuda:0')
mmdet_model7 = init_detector(mmdet_config7, mmdet_checkpoint7, device='cuda:0')
mmdet_model8 = init_detector(mmdet_config8, mmdet_checkpoint8, device='cuda:0')
mmdet_model9 = init_detector(mmdet_config9, mmdet_checkpoint9, device='cuda:0')
mmdet_model10 = init_detector(mmdet_config10, mmdet_checkpoint10, device='cuda:0')

yolo_model1 = YOLO(yolo_checkpoint1)
import yolov9
yolo_model2 = yolov9.load(
    yolov9_checkpoint1,
    device="cuda:0",
)
# SETTING
VISUALIZE = False
mmdet_threshold_base = 0.35
org_yolo_threshold = 0.5

weights = [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 15, 15, 5.5, 5.5]

iou_thr = 0.5
skip_box_thr = 0.001

# object colors
colors_dict ={'Bus':(255, 0, 0),
 'Bike': (0, 255, 0),
 'Car': (0, 0, 255),
 'Pedestrian': (255, 255, 0),
 'Truck': (255, 0, 255)}
mapping_dict = {0:'Bus', 1:'Bike', 2:'Car', 3:'Pedestrian', 4:'Truck'}

TEST_DIR = '/home/daitranskku/code/cvpr2024/aicity/DAT/CVPR_TEST/visualization/Single'
ORG_CVPR_DIR = '/home/daitranskku/code/cvpr2024/aicity/final_preprocessing_src/CPVR_test_nafnet'
test_image_file_names = os.listdir(TEST_DIR)
# extract .png only
test_image_file_names = [f for f in test_image_file_names if f.endswith('.png')]
test_image_file_names = sorted(test_image_file_names)
SUBMISSION = []
# Inference
print('Start inference...')
for test_image_file_name in tqdm(test_image_file_names):

# ##########################################################################
#     test_image_file_name = random.choice(test_image_file_names)
# ##########################################################################

    if "N" in test_image_file_name:
        mmdet_threshold = mmdet_threshold_base/2
        print("Night time")
    else:
        mmdet_threshold = mmdet_threshold_base

    image_path = os.path.join(TEST_DIR, test_image_file_name)
    image = cv2.imread(image_path)
    # inference mmdet model 1
    mmdet_results1 = inference_detector(mmdet_model1, image)
    mmdet_results2 = inference_detector(mmdet_model2, image)
    mmdet_results3 = inference_detector(mmdet_model3, image)
    mmdet_results4 = inference_detector(mmdet_model4, image)
    mmdet_results5 = inference_detector(mmdet_model5, image)
    mmdet_results6 = inference_detector(mmdet_model6, image)
    mmdet_results7 = inference_detector(mmdet_model7, image)
    mmdet_results8 = inference_detector(mmdet_model8, image)
    mmdet_results9 = inference_detector(mmdet_model9, image)
    mmdet_results10 = inference_detector(mmdet_model10, image)

    mmdet_results = [mmdet_results1, mmdet_results2, mmdet_results3, mmdet_results4, mmdet_results5, mmdet_results6, mmdet_results7, mmdet_results8, mmdet_results9, mmdet_results10]

    boxes_list = []
    scores_list = []
    labels_list = []
    # Processing MMDet models
    for mmdet_result in mmdet_results:
        raw_public_bboxes, raw_public_labels, raw_public_scores = mmdet3x_convert_to_bboxes_mmdet(mmdet_result, mmdet_threshold)
        vis_image = image.copy()
        # visualize each model's result
        temp_norm_boxes = []
        temp_scores = []
        temp_labels = []
        for i,box in enumerate(raw_public_bboxes):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # normalize for ensemble
            norm_x1 = x1 / image.shape[1]
            norm_y1 = y1 / image.shape[0]
            norm_x2 = x2 / image.shape[1]
            norm_y2 = y2 / image.shape[0]
            temp_norm_boxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
            temp_scores.append(raw_public_scores[i])
            temp_labels.append(int(raw_public_labels[i]))
        # add to list
        boxes_list.append(temp_norm_boxes)
        scores_list.append(temp_scores)
        labels_list.append(temp_labels)
    ##### Processing YOLOv8X models
    yolo_results1 = yolo_model1(image, verbose=False)
    class_names = yolo_results1[0].boxes.cls.tolist()
    boxes = yolo_results1[0].boxes.xyxy.tolist()
    scores = yolo_results1[0].boxes.conf.tolist()
    temp_norm_boxes = []
    temp_scores = []
    temp_labels = []
    for i, box in enumerate(boxes):
            if scores[i] > org_yolo_threshold:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                norm_x1 = x1 / image.shape[1]
                norm_y1 = y1 / image.shape[0]
                norm_x2 = x2 / image.shape[1]
                norm_y2 = y2 / image.shape[0]
                temp_norm_boxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
                temp_scores.append(scores[i])
                temp_labels.append(int(class_names[i]))
    boxes_list.append(temp_norm_boxes)
    scores_list.append(temp_scores)
    labels_list.append(temp_labels)
    #### Processing YOLOv9e models
    yolov9_results = yolo_model2(image_path)
    predictions = yolov9_results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    categories = [int(c) for c in categories]
    temp_norm_boxes = []
    temp_scores = []
    temp_labels = []
    for i, box in enumerate(boxes):
        if scores[i] > org_yolo_threshold:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            norm_x1 = x1 / image.shape[1]
            norm_y1 = y1 / image.shape[0]
            norm_x2 = x2 / image.shape[1]
            norm_y2 = y2 / image.shape[0]
            temp_norm_boxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
            temp_scores.append(scores[i])
            temp_labels.append(int(categories[i]))
    boxes_list.append(temp_norm_boxes)
    scores_list.append(temp_scores)
    labels_list.append(temp_labels)

    # ENSEMBLE
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    labels = [int(x) for x in labels]

    # # # # Heuristic: check overlap between pedestrian and bike, if overlap, remove bike
    for i, label in enumerate(labels):
        if label == 3:
            for j, label in enumerate(labels):
                if label == 1:
                    iou = compute_iou(boxes[i], boxes[j])
                    if iou > 0.8:
                        print('Remove bike')
                        labels[j] = -1
                        scores[j] = 0
        # check box area
        box = boxes[i]
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1*image.shape[1]), int(y1*image.shape[0]), int(x2*image.shape[1]), int(y2*image.shape[0])
        # scale
        area = (x2-x1)*(y2-y1)
        if area < 120:
            labels[i] = -1
            scores[i] = 0
            print('Remove small box')
    # check overlap between same class
    for i, box in enumerate(boxes):
        for j, box in enumerate(boxes):
            if i != j:
                if labels[i] == labels[j]:
                    iou = compute_iou(boxes[i], boxes[j])
                    if iou > 0.9:
                        print('Remove overlap')
                        if scores[i] > scores[j]:
                            labels[j] = -1
                            scores[j] = 0
                        else:
                            labels[i] = -1
                            scores[i] = 0
    # # remove -1
    boxes = [boxes[i] for i in range(len(boxes)) if labels[i] != -1]
    scores = [scores[i] for i in range(len(scores)) if labels[i] != -1]
    labels = [labels[i] for i in range(len(labels)) if labels[i] != -1]

    ### Remove box lower than threshold
    if "N" in test_image_file_name:
        box_threshold = mmdet_threshold_base/2
    else:
        box_threshold =mmdet_threshold_base
    boxes = [boxes[i] for i in range(len(boxes)) if scores[i] >= box_threshold]
    labels = [labels[i] for i in range(len(labels)) if scores[i] >= box_threshold]
    scores = [scores[i] for i in range(len(scores)) if scores[i] >= box_threshold]

    ##### Start visualize with org image
    org_image_file_name = test_image_file_name.split('/')[-1].split('.')[0][:-3] + '.png'
    print(org_image_file_name)
    org_image_path = os.path.join(ORG_CVPR_DIR, org_image_file_name)
    vis_image = cv2.imread(org_image_path)
    if VISUALIZE:
        for i,box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1*vis_image.shape[1]), int(y1*vis_image.shape[0]), int(x2*vis_image.shape[1]), int(y2*vis_image.shape[0])
            detected_class = mapping_dict[labels[i]]
            color = colors_dict[detected_class]
            # Visualize
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            # show score
            cv2.putText(vis_image, str(np.round(scores[i],2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)
        ## Visualize prediction and GT
        # # # ##### VISUALIZE
        cv2.imwrite('cvpr_submission.png', vis_image)
        scale_x = 800
        scale_y = 800
        vis_image = cv2.resize(vis_image, (scale_x, scale_y))
        cv2.imshow('Prediction', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # add to submission
    temp_submission = []
    for i,box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1*vis_image.shape[1]), int(y1*vis_image.shape[0]), int(x2*vis_image.shape[1]), int(y2*vis_image.shape[0])
        detected_class = mapping_dict[labels[i]]
        score = scores[i]
        temp_submission.append({
            "image_id": get_image_Id(org_image_file_name),
            "category_id": int(labels[i]),
            "bbox": [x1, y1, x2-x1, y2-y1],
            "score": score
        })
    SUBMISSION.extend(temp_submission)
    # break
# Save submission
current_dir = os.getcwd()
print('Save submission to: {}'.format(current_dir))
with open('cvpr_submission.json', 'w') as f:
    json.dump(SUBMISSION, f)