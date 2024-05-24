import json
import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np
def generate_final_images(org_json, DAY_DIR, day_image_names, NAFNET_DIR):
    total_image_info = []
    total_annotation_info = []
    num_change = 0
    for raw_image_info in org_json['images']:
        image_path = raw_image_info['file_name']
        extracted_name = image_path.split('/')[-1]
        if extracted_name in day_image_names:
            day_image_path = os.path.join(DAY_DIR, extracted_name)
            image_info = {}
            image_info['id'] = raw_image_info['id']
            image_info['width'] = raw_image_info['width']
            image_info['height'] = raw_image_info['height']
            image_info['file_name'] = day_image_path
            total_image_info.append(image_info)
            num_change += 1
        else:
            day_image_path = os.path.join(NAFNET_DIR, extracted_name)
            image_info = {}
            image_info['id'] = raw_image_info['id']
            image_info['width'] = raw_image_info['width']
            image_info['height'] = raw_image_info['height']
            image_info['file_name'] = day_image_path
            total_image_info.append(image_info)
    save_json = {}
    save_json['images'] = total_image_info
    save_json['annotations'] = org_json['annotations']
    save_json['categories'] = org_json['categories']
    return save_json

VISUALIZE = True
print("GENERATE FINAL TRAIN JSON")
org_json_path = '/home/daitranskku/code/cvpr2024/aicity/AIC2024-TRACK4-TEAM15/train_raw_fisheye8k.json'
org_json = json.load(open(org_json_path))
DAY_DIR = '/home/daitranskku/code/cvpr2024/aicity/AIC2024-TRACK4-TEAM15/sample_dataset/GSAD_Output/train'
day_image_names = os.listdir(DAY_DIR)
NAFNET_DIR = '/home/daitranskku/code/cvpr2024/aicity/AIC2024-TRACK4-TEAM15/sample_dataset/NAFNet_Output/train'
json_data = generate_final_images(org_json, DAY_DIR, day_image_names, NAFNET_DIR)
if VISUALIZE:
    image_info = random.choice(json_data['images'])
    image_file_path = image_info['file_name']
    image = cv2.imread(image_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            # print(category_id)
            # print(bbox)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            image = cv2.putText(image, str(category_id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
                                4)
    plt.imshow(image)
    plt.show()
with open("nafnet_train_night2day.json", "w") as train_file:
    json.dump(json_data, train_file)

print("GENERATE FINAL VAL JSON")
org_json_path = '/home/daitranskku/code/cvpr2024/aicity/AIC2024-TRACK4-TEAM15/val_raw_fisheye8k.json'
org_json = json.load(open(org_json_path))
DAY_DIR = '/home/daitranskku/code/cvpr2024/aicity/AIC2024-TRACK4-TEAM15/sample_dataset/GSAD_Output/val'
day_image_names = os.listdir(DAY_DIR)
NAFNET_DIR = '/home/daitranskku/code/cvpr2024/aicity/AIC2024-TRACK4-TEAM15/sample_dataset/NAFNet_Output/val'
json_data = generate_final_images(org_json, DAY_DIR, day_image_names, NAFNET_DIR)
if VISUALIZE:
    image_info = random.choice(json_data['images'])
    image_file_path = image_info['file_name']
    image = cv2.imread(image_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            # print(category_id)
            # print(bbox)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            image = cv2.putText(image, str(category_id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
                                4)
    plt.imshow(image)
    plt.show()
with open("nafnet_val_night2day.json", "w") as train_file:
    json.dump(json_data, train_file)