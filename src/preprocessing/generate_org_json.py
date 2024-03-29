import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import natsort
from tqdm import tqdm
import random
import numpy as np
import json

DIR = '/home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/FishEye8K'
TRAIN_FOLDERS =  ['train/images']
VAL_FOLDERS = ['val/images']

TRAIN_ANNOTATION_PATH = '/home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/FishEye8K/train/labels'
VAL_ANNOTATION_PATH = '/home/daitranskku/code/cvpr2024/aicity/github_submission/sample_dataset/FishEye8K/val/labels'

VISUALIZE = True

class_name_to_id = {
    "Bus": 0,
    "Bike": 1,
    "Car": 2,
    "Pedestrian": 3,
    "Truck": 4,
}
colors_dict ={'Bus':(255, 0, 0),
 'Bike': (0, 255, 0),
 'Car': (0, 0, 255),
 'Pedestrian': (255, 255, 0),
 'Truck': (255, 0, 255)}
mapping_dict = {0:'Bus', 1:'Bike', 2:'Car', 3:'Pedestrian', 4:'Truck'}

def extract_class_and_bounding_box(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    extracted_data = []
    for obj in root.iter('object'):
        obj_class = obj.find('name').text
        bndbox = obj.find('bndbox')
        bounding_box = {
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        }
        extracted_data.append((obj_class, bounding_box))
    return extracted_data

def generate_json(TRAIN_FOLDERS, TRAIN_ANNOTATION_PATH):
    image_id = 1
    annotation_id = 1
    total_image_info = []
    total_annotation_info = []
    for folder in tqdm(TRAIN_FOLDERS):
        folder_path = os.path.join(DIR, folder)
        image_file_names = os.listdir(folder_path)
        image_file_names = [x for x in image_file_names if x[-4:] == '.png']
        for image_file_name in tqdm(image_file_names):
            image_path = os.path.join(folder_path, image_file_name)
            image = cv2.imread(image_path)
            # add image info
            image_info = {}
            image_path = image_path
            image_info['id'] = image_id
            image_id += 1
            image_info['width'] = image.shape[1]
            image_info['height'] = image.shape[0]
            image_info['file_name'] = image_path
            total_image_info.append(image_info)
            # extract annotation file
            annotation_file_name = image_file_name[:-4] + '.xml'
            annotation_path = os.path.join(TRAIN_ANNOTATION_PATH, annotation_file_name)
            # read xml
            extracted_data = extract_class_and_bounding_box(annotation_path)
            # visualize
            for obj_class, bounding_box in extracted_data:
                xmin = bounding_box['xmin']
                ymin = bounding_box['ymin']
                xmax = bounding_box['xmax']
                ymax = bounding_box['ymax']
                x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin
                # add annotation info
                annotation_info = {}
                annotation_info['id'] = annotation_id
                annotation_id += 1
                annotation_info['image_id'] = image_info['id']
                annotation_info['category_id'] = class_name_to_id[str(obj_class)]
                annotation_info['bbox'] = [x, y, w, h]
                annotation_info["segmentation"] = [[x,y,x+w,y,x+w,y+h,x,y+h]]
                annotation_info['area'] = w*h
                annotation_info['iscrowd'] = 0
                total_annotation_info.append(annotation_info)
    return total_image_info, total_annotation_info

def visualize_json(json_data):
    image_info = random.choice(json_data['images'])
    image_file_path = image_info['file_name']
    image = cv2.imread(image_file_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # read label
    image_id = image_info['id']
    for annotation in json_data['annotations']:
        if annotation['image_id'] == image_id:
            # print("Found annotation")
            bbox = annotation['bbox']
            bbox = [int(x) for x in bbox]
            bbox = np.array(bbox)
            bbox = bbox.astype(np.int32)
            category_id = annotation['category_id']
            detected_class = mapping_dict[int(category_id)]
            color = colors_dict[detected_class]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
            cv2.putText(image, detected_class, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

total_category_info = []
for check in class_name_to_id:
    category_info = {}
    category_info['id'] = class_name_to_id[check]
    category_info['name'] = check
    category_info['supercategory'] = 'root'
    total_category_info.append(category_info)

## Generate train json
print('Generate train json')
total_image_info, total_annotation_info = generate_json(TRAIN_FOLDERS, TRAIN_ANNOTATION_PATH)
save_json = {}
save_json['images'] = total_image_info
save_json['annotations'] = total_annotation_info
save_json['categories'] = total_category_info
json_data = save_json
if VISUALIZE:
    # VISUALIZE
    visualize_json(json_data)
## Save json
with open("train_raw_fisheye8k.json", "w") as train_file:
    json.dump(json_data, train_file)
current_dir = os.getcwd()
print("Save train json at: {}".format(os.path.join(current_dir, "train_raw_fisheye8k.json")))
## Generate val json
print('Generate val json')
total_image_info, total_annotation_info = generate_json(VAL_FOLDERS, VAL_ANNOTATION_PATH)
save_json = {}
save_json['images'] = total_image_info
save_json['annotations'] = total_annotation_info
save_json['categories'] = total_category_info
json_data = save_json
if VISUALIZE:
    # VISUALIZE
    visualize_json(json_data)
## Save json
with open("val_raw_fisheye8k.json", "w") as val_file:
    json.dump(json_data, val_file)
print("Save val json at: {}".format(os.path.join(current_dir, "val_raw_fisheye8k.json")))












