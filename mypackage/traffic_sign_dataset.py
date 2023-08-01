import os
import random

import pandas as pd
from collections import defaultdict, namedtuple
import csv

import cv2
from PIL import Image

from torch.utils.data import Dataset, DataLoader, Subset

import torchvision
import torchvision.utils
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import seaborn as sns


class To_Dataset(Dataset):
    """Custom PyTorch Dataset class to load image data and labels.

    Args:
        images (list): A list containing image data (e.g., image paths or image tensors).
        targets (list): A list containing target labels for each image.
        classes (list): A list of class names or class labels associated with the dataset.

    Attributes:
        class_to_idx (dict): A dictionary that maps class names or labels to their corresponding indices.

    Note:
        The images and targets lists must be of the same length.

    Example:
        images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
        targets = [0, 1, 0]
        classes = ['cat', 'dog']

        dataset = To_Dataset(images, targets, classes)
    """
    def __init__(self, images, targets, classes):
        self.images = images
        self.targets = targets
        self.classes = classes
        self.class_to_idx = {'00000': 0,'00001': 1,'00002': 2,'00003': 3,'00004': 4,'00005': 5,'00006': 6,
                             '00007': 7,'00008': 8,'00009': 9,'00010': 10,'00011': 11,'00012': 12,
                             '00013': 13,'00014': 14,'00015': 15,'00016': 16,'00017': 17,'00018': 18,
                             '00019': 19,'00020': 20,'00021': 21,'00022': 22,'00023': 23,'00024': 24,
                             '00025': 25,'00026': 26,'00027': 27,'00028': 28,'00029': 29,'00030': 30,
                             '00031': 31,'00032': 32,'00033': 33,'00034': 34,'00035': 35,'00036': 36,
                             '00037': 37,'00038': 38,'00039': 39,'00040': 40,'00041': 41,'00042': 42,
                             '00043': 43,'00044': 44,'00045': 45,'00046': 46,'00047': 47,'00048': 48,
                             '00049': 49,
                            }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
        return image, target


class TrafficSignDataset(Dataset):
    """Custom PyTorch Dataset class to load and preprocess the traffic sign dataset.

    Args:
        data_dir (str): The directory path to the dataset containing training images.
        transform (callable, optional): A callable object that applies transformations to the images (default: None).
        test_labels_path (str, optional): The path to the CSV file containing annotations for test data (default: None).
        ann_filename_row (int, optional): The row index in the test annotation CSV where filenames are present (default: 0).
        ann_labels_row (int, optional): The row index in the test annotation CSV where labels are present (default: 3).

    Attributes:
        class_to_idx (dict): A dictionary that maps class names or labels to their corresponding indices.

    Note:
        - If `test_labels_path` is None, the dataset is assumed to be a training dataset.
        - If `test_labels_path` is provided, the dataset is assumed to be a test dataset.
    """

    def __init__(self, data_dir, transform=None, test_labels_path=None, ann_filename_row=0, ann_labels_row=3):
        """Initialize the TrafficSignDataset.

        Args:
            data_dir (str): The directory path to the dataset containing training images.
            transform (callable, optional): A callable object that applies transformations to the images (default: None).
            test_labels_path (str, optional): The path to the CSV file containing annotations for test data (default: None).
            ann_filename_row (int, optional): The row index in the test annotation CSV where filenames are present (default: 0).
            ann_labels_row (int, optional): The row index in the test annotation CSV where labels are present (default: 3).
        """
        # set directories
        self.data_dir = data_dir
        self.test_labels_path = test_labels_path
        
        # set the position of the filename and label-Id the annotation csv of the test data 
        self.ann_filename_row = ann_filename_row
        self.ann_labels_row = ann_labels_row
        
        # lists for training data
        self.images = []
        self.targets = []
        
        self.transform = transform # augmentation
        # labels dictionary
        self.labels_dictionary = {
            0: "speed limit 20 (prohibitory)",1: "speed limit 30 (prohibitory)",2: "speed limit 50 (prohibitory)",
            3: "speed limit 60 (prohibitory)",4: "speed limit 70 (prohibitory)",5: "speed limit 80 (prohibitory)",
            6: "restriction ends 80 (other)",7: "speed limit 100 (prohibitory)",8: "speed limit 120 (prohibitory)",
            9: "no overtaking (prohibitory)",10: "no overtaking (trucks) (prohibitory)",
            11: "priority at next intersection (danger)", 12: "priority road (other)",
            13: "give way (other)",                       14: "stop (other)",
            15: "no traffic both ways (prohibitory)",     16: "no trucks (prohibitory)",
            17: "no entry (other)",                       18: "danger (danger)",
            19: "bend left (danger)",                     20: "bend right (danger)",
            21: "bend (danger)",                          22: "uneven road (danger)",
            23: "slippery road (danger)",                 24: "road narrows (danger)",
            25: "construction (danger)",                  26: "traffic signal (danger)",
            27: "pedestrian crossing (danger)",           28: "school crossing (danger)",
            29: "cycles crossing (danger)",               30: "snow", 31: "animals", 32: "restriction ends",
            33: "go right", 34: "go left",   35: "go straight",       36: "go right or straight",
            37: "go left or straight",       38: "keep right mandatory",
            39: "keep left (mandatory)",     40: "roundabout (mandatory)",
            41: "restriction ends (overtaking) (other)",  42: "restriction ends (overtaking (trucks)) (other)",
            43: "one-way street left",                    44: "one-way street right",
            45: "bus stop", 46: "bike path", 47: "absolute stopping ban",            48: "restricted parking ban",
            49: "pedestrian crossing",
        }
        
        # get dataset
        self.dataset = self.load_dataset() # load train/test dataset

        self.class_to_idx = self.dataset.class_to_idx


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label
    
    def load_dataset(self):
        """
        Load the dataset either from the training directory or from the test images and annotations.

        Returns:
            torchvision.datasets.ImageFolder or To_Dataset: The loaded dataset, either as an ImageFolder (for training data)
            or as a custom To_Dataset (for test data).
        """
        # load train data
        if not self.test_labels_path:
            dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        
        # load test data
        elif self.test_labels_path:    
            test_images = []
            p = os.listdir(self.data_dir)
            p.sort()
            for i, filename in enumerate(p):
                img = cv2.imread(os.path.join(self.data_dir,filename)) # read images
                if img is not None:
                    if self.transform:
                        img = self.transform(img)
                    test_images.append(img)            
            
            test_annotations = self.read_annotations() # Read test annotations
            
            test_labels = [t[1] for t in test_annotations] # extract test labels
            
            dataset = To_Dataset(test_images, test_labels, self.labels_dictionary)
            
        for image, label in dataset:
            self.images.append(image)
            self.targets.append(label)
            
        return dataset
    
    
    def read_annotations(self):
        """
        Read and parse annotations from the CSV file.

        Returns:
            list: A list of named tuples representing annotations, where each tuple contains the filename and the label.
        """
        Annotation = namedtuple('Annotation', ['filename', 'label'])
        annotations = [] 
        with open(self.test_labels_path) as f:
            reader = csv.reader(f, delimiter=';')
            next(reader) # skip header

            # loop over all images in current annotations file
            for row in reader:
                filename = row[self.ann_filename_row] # filename is in the 0th column
                label = int(row[self.ann_labels_row]) # label is in the 7th column
                annotations.append(Annotation(filename, label))

        return annotations       
        

def get_data_loader(dataset, batch_size, shuffle=True):
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to create the DataLoader from.
        batch_size (int): The batch size to use for the DataLoader.
        shuffle (bool, optional): Whether to shuffle the data during loading. Default is True.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the given dataset.
    """
    
    k = len(dataset) % batch_size
    data_set = Subset(dataset, range(k, len(dataset)))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader
