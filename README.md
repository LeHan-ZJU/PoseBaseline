# PoseBaseline
## introduction
This is the basic code of the 2D pose estimation model I built.

```
PoseBaseline
├── train.py   // Model training code
├── Models
│   │── RatNet.py  // model_bulid
│   └── eval_pose.py  // validation
├── utils
│   └── dataset_csv.py  // It read image and its labels in csv file, and generate heatmaps based on the labels. Then it convert data into tensor and return. This is the basic csv file reading and image processing, you can read and modify according to your own needs.
```
