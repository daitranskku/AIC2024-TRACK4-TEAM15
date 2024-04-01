[CVPRW 2024] Low-Light Image Enhancement Framework for Improved Object Detection in Fisheye Lens Datasets
================================================================================================
## Introduction
This repo contains **Team 15 (SKKU-NDSU)** code for the **Track 4** submission  to the [CVPR AI City Challenge 2024](https://www.aicitychallenge.org/). 
We propose a data preprocessing framework called the **Low-Light Image Enhancement Framework**. This framework utilizes a transformer-based image enhancement technique, NAFNet, to increase image clarity by removing blurriness and the use of GSAD to convert nighttime images (low illumination) to daytime images (high illumination) to improve accuracy in object detection for fisheye images during model training. To further improve the accuracy of object detection during inference, the study employed a super-resolution postprocessing technique, DAT, to increase the pixels of the images for enhanced object detection, as well as an ensemble model technique for robust detection.
## Proposed Approach
Our methodology is visually encapsulated in the figures below, demonstrating our comprehensive approach and exemplary results.
![figure](./figures/proposed_approach.png)

![figure2](./figures/sample_results.png)
## Installation
Here is the list of libraries used in this project:
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [NAFNet](https://github.com/megvii-research/NAFNet)
- [GSAD](https://github.com/jinnh/GSAD)
- [DAT](https://github.com/zhengchen1999/DAT)
- [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9)

Note: For optimal performance, we recommend setting up separate environments for each library.
The proposed approach is inference on Intel Core i9, and NVIDIA 4090 24GB and 64GB RAM. Models are trained on Intel Xeon Silver 4210R, and 2 NVIDIA RTX A6000 48GB and 126GB RAM
## Preprocessing
### Download the FishEye8K Dataset
Download the FishEye8K dataset from the [AI City Challenge](https://github.com/MoyoG/FishEye8K).
Ensure the following directory structure for seamless integration:
```
FishEye8K
│
└───train
│   │
│   └───images
│   │   
│   └───labels
│
└───val
│   │
│   └───images
│   │
│   └───labels
│
└───test
│   │
│   └───images
```
### Generate JSON Annotation Files
Modify the following paths in [generate_org_json.py](./src/preprocessing/generate_org_json.py) to generate the JSON annotation files:
- DIR
- TRAIN_ANNOTATION_PATH 
- VAL_ANNOTATION_PATH 

_Note: we used absolute paths in the code. Please modify the paths accordingly._
```
python src/preprocessing/generate_org_json.py
```
### Image Enhancement using NAFNet
We directly use the pre-trained model **NAFNet-REDS-width64** from the [NAFNet](https://github.com/megvii-research/NAFNet) official GitHub repo.
```
python src/lib/NAFNet/nafnet_inference.py
```
### Convert to day-light images using GSAD
Pre-trained model [GSAD](https://github.com/jinnh/GSAD) is used to convert the night images to day-like images. Pre-trained model on the LOLv2Syn dataset is used.
```
sh src/lib/GSAD/convert_day_like.sh
```
### Generate the final dataset
```
python src/preprocessing/generate_final_dataset.py
```
## Training Co-DETR
Here we show how to train the CO-DETR model on the FishEye8K dataset. 
Other models such as [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9) are trained based on their original GitHub repositories.
```
sh tools/dist_train.sh projects/CO-DETR/configs/codino/swinL_detr_o365_coco.py 2
```
## Super Resolution
We use the [DAT](https://github.com/zhengchen1999/DAT) algorithm to perform super-resolution on the testing images. Each test image is scaled by a factor of 4.
```
cd src/lib/DAT
python basicsr/test.py -opt options/Test/convert_CVPR_test.yaml
```
## Inference for submission on CVPR test set
Download the super-resolution test images in [here](https://drive.google.com/drive/folders/1mqTm7k5I1S1lBULDyg2hn5Ybp5KvM5KI?usp=sharing). 

Download all pre-trained models in [here](https://drive.google.com/drive/folders/1r6YZBC8Z8moq7nrVH7UVj1JxFhMfFTuN?usp=drive_link)
```
python src/inference4submission.py
```
## Contact
If you have any questions, please feel free to contact Dai Tran [(daitran@skku.edu)](daitran@skku.edu).




