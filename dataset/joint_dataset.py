import json
import os
import csv
import torch.utils.data as data
import torch
import numpy as np
import pandas as pd
import nltk

class ShapeCaptionDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 16
    POINT_DIMENSION = 3

    def __init__(self, shapes_dir, captions_csv, vocab, num_of_points, train=True):
        self.shapes_dir = shapes_dir
        self.captions_csv  = captions_csv
        self.vocab = vocab
        self.number_of_points = num_of_points
        # loading captions in memory
        self.captions = pd.read_csv(captions_csv)

        category_file = os.path.join(self.dataset_folder, 'synsetoffset2category.txt')
        self.folders_to_classes_mapping = {}
        self.segmentation_classes_offset = {}

        with open(category_file, 'r') as fid:
            reader = csv.reader(fid, delimiter='\t')
            offset_seg_class = 0
            for k, row in enumerate(reader):
                self.folders_to_classes_mapping[row[1]] = k
                self.segmentation_classes_offset[row[1]] = offset_seg_class
                # offset_seg_class += self.PER_CLASS_NUM_SEGMENTATION_CLASSES[row[0]]

        if self.train:
            filelist = os.path.join(self.shapes_dir, 'train_test_split', 'shuffled_train_file_list.json')
        else:
            filelist = os.path.join(self.shapes_dir, 'train_test_split', 'shuffled_test_file_list.json')

        with open(filelist, 'r') as fid:
            filenames = json.load(fid)

        #TODO: filter table and chair categories only
        self.files = [(f.split('/')[1], f.split('/')[2]) for f in filenames]

    def __getitem__(self, index):

        caption = self.captions[index]
        model_id = caption['model_id']
        sh_folder, sh_file = self.files[model_id]
        point_file = os.path.join(self.shapes_dir,
                                  sh_folder,
                                  'points',
                                  '%s.pts' % sh_file)

        point_cloud_class = self.folders_to_classes_mapping[sh_folder]
        points = self.prepare_shape_data(point_file,
                                         self.number_of_points,
                                         point_cloud_class=point_cloud_class)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return points, target

    def __len__(self):
        return len(self.captions.index)

    @staticmethod
    def prepare_shape_data(point_file,
                     number_of_points=None,
                     point_cloud_class=None,
                     segmentation_label_file=None,
                     segmentation_classes_offset=None):
        point_cloud = np.loadtxt(point_file).astype(np.float32)
        if number_of_points:
            sampling_indices = np.random.choice(point_cloud.shape[0], number_of_points)
            point_cloud = point_cloud[sampling_indices, :]
        point_cloud = torch.from_numpy(point_cloud)
        if segmentation_label_file:
            segmentation_classes = np.loadtxt(segmentation_label_file).astype(np.int64)
            if number_of_points:
                segmentation_classes = segmentation_classes[sampling_indices]
            segmentation_classes = segmentation_classes + segmentation_classes_offset - 1
            segmentation_classes = torch.from_numpy(segmentation_classes)
            return point_cloud, segmentation_classes
        elif point_cloud_class is not None:
            point_cloud_class = torch.tensor(point_cloud_class)
            return point_cloud, point_cloud_class
        else:
            return point_cloud


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

        We should build custom collate_fn rather than using default collate_fn,
        because merging caption (including padding) is not supported in default.

        Args:
            data: list of tuple (image, caption).
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
    # Sort a data list by caption length (descending order).
    #TODO: don't know if reverse should be True or not?
    data.sort(key=lambda x: len(x[1]), reverse=True)
    shapes, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    shapes = torch.stack(shapes, 0)

    # building a batch for captions
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return shapes, targets, lengths

def get_loader(shapes_dir, captions_csv, vocab, num_of_points, train,
               batch_size, shuffle, num_workers):
    t2s_dataset = ShapeCaptionDataset(shapes_dir=shapes_dir,
                                      captions_csv=captions_csv,
                                      vocab=vocab,
                                      num_of_points=num_of_points,
                                      train=train)

    data_loader = torch.utils.data.DataLoader(dataset=t2s_dataset,
                                              batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader


