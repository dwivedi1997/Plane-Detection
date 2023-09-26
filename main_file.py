import os
import pandas as pd
from dataset_download import kaggle_api
import numpy as np

## Downloading Datasets
link = "https://www.kaggle.com/datasets/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes/download?datasetVersionNumber=1"
data = kaggle_api(link)
#data.download()

train_img_dir = "cgi-planes-in-satellite-imagery-w-bboxes/training/train"
test_img_dir = "cgi-planes-in-satellite-imagery-w-bboxes/training/test"

train_csv_dir = "cgi-planes-in-satellite-imagery-w-bboxes/train_labels.csv"
test_csv_dir = "cgi-planes-in-satellite-imagery-w-bboxes/test_labels.csv"

from process_file import file_process

file = file_process(train_img_dir, test_img_dir, train_csv_dir, test_csv_dir)
file.image_name()
train_img = file.train_img
test_img = file.test_img

print('Train Images - {}'.format(len(train_img)))
print('Test Images - {}'.format(len(test_img)))

## Make Directory
os.mkdir('train_data')
os.mkdir('train_data/images')
os.mkdir('train_data/images/train')
os.mkdir('train_data/images/val')
os.mkdir('train_data/labels')
os.mkdir('train_data/labels/train')
os.mkdir('train_data/labels/val')

train_img_opdir = "train_data/images/train"
test_img_opdir = "train_data/images/val"

train_labels_opdir = "train_data/labels/train"
test_labels_opdir = "train_data/labels/val"

file.copy(train_img_opdir, test_img_opdir)

train_csv = pd.read_csv(train_csv_dir)
test_csv = pd.read_csv(test_csv_dir)
train_csv.head()

class_names = train_csv["class"].unique()

class_mapping = {}
for i, class_name in enumerate(class_names):
    class_mapping[i] = class_name

idx_mapping = {}
for idx, class_name in class_mapping.items():
    idx_mapping[class_name] = idx

print(class_mapping)
print(idx_mapping)

from image_display import image_visualization

## Looping through train and test image and saving labels in the YOLOV5 tree format
for image_no, image_name in enumerate(train_img):

    image_labels = train_csv[train_csv["filename"]==image_name]
    image_id = image_name.split('.')[0]

    labels = []
    bounding_boxes = []
    image_sizes= []
    for _, row in image_labels.iterrows():

        _, img_width, img_height, class_name, xmin, ymin, xmax, ymax = row.to_numpy()

        class_id   = idx_mapping[class_name]
        label = class_mapping[class_id]
        box_width  = xmax-xmin
        box_height = ymax-ymin
        x_center   = xmin + (box_width/2)
        y_center   = ymin + (box_height/2)

        x_center /= img_width
        box_width /= img_width
        y_center /= img_height
        box_height /= img_height

        single_label = np.array([class_id, x_center, y_center, box_width, box_height])

        single_label = np.expand_dims(single_label, axis=0)
        bounding_boxes.append(single_label)
        image_sizes.append((img_width,img_height))
        labels.append(label)

    bounding_boxes = np.concatenate(bounding_boxes, axis=0)

    i = 55
    if image_no==i:
        print(image_name)
        print(labels)
        print(bounding_boxes)
        img=image_visualization(train_img_dir, image_name, bounding_boxes, img_width, img_height,labels)
        img.visualize()

    SAVE_DIR = os.path.join(train_labels_opdir, f"{image_id}.txt")
    np.savetxt(SAVE_DIR, bounding_boxes, fmt=['%d', '%f', '%f', '%f', '%f'])

for image_no, image_name in enumerate(test_img):

    image_labels = test_csv[test_csv["filename"]==image_name]
    image_id = image_name.split('.')[0]

    labels = []
    bounding_boxes = []
    image_sizes= []
    for _, row in image_labels.iterrows():

        _, img_width, img_height, class_name, xmin, ymin, xmax, ymax = row.to_numpy()

        class_id   = idx_mapping[class_name]
        label = class_mapping[class_id]
        box_width  = xmax-xmin
        box_height = ymax-ymin
        x_center   = xmin + (box_width/2)
        y_center   = ymin + (box_height/2)

        x_center /= img_width
        box_width /= img_width
        y_center /= img_height
        box_height /= img_height

        single_label = np.array([class_id, x_center, y_center, box_width, box_height])

        single_label = np.expand_dims(single_label, axis=0)
        bounding_boxes.append(single_label)
        image_sizes.append((img_width,img_height))
        labels.append(label)

    bounding_boxes = np.concatenate(bounding_boxes, axis=0)

    i = 55
    if image_no==i:
        print(image_name)
        print(labels)
        print(bounding_boxes)
        img=image_visualization(test_img_dir, image_name, bounding_boxes, img_width, img_height,labels)
        img.visualize()

    SAVE_DIR = os.path.join(test_labels_opdir, f"{image_id}.txt")
    np.savetxt(SAVE_DIR, bounding_boxes, fmt=['%d', '%f', '%f', '%f', '%f'])
