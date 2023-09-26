import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import cv2
import os
import shutil


class file_process():

    def __init__(self, train_img_dir, test_img_dir, train_csv_dir, test_csv_dir):
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.train_csv_dir = train_csv_dir
        self.test_csv_dir = test_csv_dir

    ## Fetching Image name
    def image_name(self):
        self.train_img = []
        for img in os.listdir(self.train_img_dir):
            if img.endswith(".png"):
                self.train_img.append(img)

        self.test_img=[]
        for img in os.listdir(self.test_img_dir):
            if img.endswith(".png"):
                self.test_img.append(img)

    def copy(self, to_train, to_test):
        for image_name in self.train_img:
            from_location = os.path.join(self.train_img_dir, image_name)
            to_location = os.path.join(to_train, image_name)
            shutil.copy(from_location, to_location)

        for image_name in self.test_img:
            from_location = os.path.join(self.test_img_dir, image_name)
            to_location = os.path.join(to_test, image_name)
            shutil.copy(from_location, to_location)

