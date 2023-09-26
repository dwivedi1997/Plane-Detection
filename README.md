# Object-Detection-YOLO_v5
Problem Statement : <b>Detection of Plane using Yolo_v5 by Ultralytics</b>
## Overview
For this project, I am using the YOLOv5 to train an object detection model. YOLO is an acronym for <b>“You Only Look Once”</b>. Below are the salient features:

1. <b>Speed</b> (Base model — 45 frames per second, Fast model — 155 frames per second, 1000x faster than R-CNN, ) <br>
2. The architecture comprises only a single neural network (Can be optimized end to end directly on detection performance)<br>
3. Able to learn the general representation of objects (Predictions are informed by the global context of image)
Open-source.

## Available Models
There are a total of 4 models to train. You can select any one of them during training. They are as follows:<br>

1. <b>YOLOv5s</b> - yolov5 small, about 7 million parameter, file - yolov5s.yaml<br>
2. <b>YOLOv5m</b> - yolov5 medium, about 21 million parameters, file - yolo5m.yaml<br>
3. <b>YOLOv5l</b> - yolov5 large, about 47 million parameters, file - yolo5l.yaml<br>
4. <b>YOLOv5x</b> - yolov5 extra-large, about 87 million parameters, file - yolo5x.yaml<br>


## GPU
The model was train on <b>Nvidia GPU A4000</b> provided by the <a href='www.paperspace.com'>Paperspace</a>.


## Datasets
Two folders of 500 computer generated satellite images of planes accompanied with an xml of the bounding box coordinates.
The dataset can be download here: <a href='https://www.kaggle.com/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes'> Datasets </a>


## Test Images
![image](https://github.com/dwivedi1997/Plane-Detection/assets/47722937/5aa74214-421c-4f09-b0e3-34e8a8d4c814)

## Inferences
Results
![image](https://github.com/dwivedi1997/Plane-Detection/blob/master/results/results.png?raw=true)

<b> Confusion Matrix </b>
![image](https://github.com/dwivedi1997/Plane-Detection/blob/master/results/results.png?raw=true)
