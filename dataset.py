import json
import os
import csv
import torch.utils.data as data
import torch
import numpy as np

class ShapeNetDataSet(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 16
    NUM_SEGMENTATION_CLASSES = 50
    POINT_DIMENSION = 3

    def __init__(self, dataset_dir, num_of_points=2048, train=True):
        self.dataset_folder = dataset_dir
        self.number_of_points = num_of_points
        self.train = train

        category_file = os.path.join(self.dataset_folder, 'synsetoffset2category.txt')

        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            offset_seg_class = 0
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[1]] = k
                self.segmentation_classes_offset[row[1]] = offset_seg_class
                offset_seg_class += self.PER_CLASS_NUM_SEGMENTATION_CLASSES[row[0]]

        if self.train:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_train_file_list.json')
        else:
            filelist = os.path.join(self.dataset_folder, 'train_test_split', 'shuffled_test_file_list.json')

        with open(filelist, 'r') as fid:
            filenames = json.load(fid)

        self.files = [(f.split('/')[1], f.split('/')[2]) for f in filenames]
