# PoseBaseline
## introduction
This is the pytorch implementation of 2D pose estimation model. 

The model mainly consists of resnet50 and three up-sampling modules. 
The output of the model is a set of heatmaps of c*(w/4)*(h/4), where c is the number of keypoints, and w and h are the dimensions of the input images.

## Structure
```
PoseBaseline
├── train.py   // Model training code
├── Models
│   │── RatNet.py  // model_bulid
│   └── eval_pose.py  // validation
├── utils
│   └── dataset_csv.py  // It read image and its labels in csv file, and generate heatmaps based on the labels. Then it convert data into tensor and return. This is the basic csv file reading and image processing, you can read and modify according to your own needs.
```
