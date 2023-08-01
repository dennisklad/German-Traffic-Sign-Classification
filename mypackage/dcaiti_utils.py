#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


def show(img):
    """
    This method takes an image in numpy format and displays it.

    Parameters
    ----------
    img : a numpy array describing an image

    Returns
    -------
    None.
    """
    shape = img.shape

    if len(shape) == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()
    if len(shape) == 2:
        plt.imshow(img, cmap='gray')
        plt.show()

def show_np(img):
    """
    This method takes an image in numpy format and displays it with addtitional transformation to PIL image
    """
    
    img = np.asarray(transforms.ToPILImage()(img))
    show(img)
    
        
def load_images_from_folder(path, transform=None):
    """
    This method takes a folder path and reads image files with the specified transformation.

    Parameters
    ----------
    path : str
        A string representing the path to the folder containing image files.
    transform : torchvision.transforms.Compose
        A composition of image transformations from torchvision.transforms.

    Returns
    -------
    list
        A list of transformed image tensors.

    """
    
    images = []
    p = os.listdir(path)
    p.sort() # sort the list of images
    for i, filename in enumerate(p):
        img = cv2.imread(os.path.join(path,filename))
        
        if transform:
            img = transform(img)
        
        if img is not None:
            images.append(img)
            
    print(f"{i} images were read")
        
    return images


def resize_image(image, target_size):
    """
    This method takes an image and a target size and resizes the image while preserving its aspect ratio.

    Parameters
    ----------
    image : a numpy array describing an image
    target_size : an integer describing the desired size

    Returns
    -------
    numpy array
        The resized image.

    """
    height, width = image.shape[:2]
    aspect_ratio_image = width / height
    aspect_ratio_target = 1

    # Determine the scale factor based on the aspect ratios
    if aspect_ratio_image > aspect_ratio_target:
        scale = target_size / width
    else:
        scale = target_size / height

    # Calculate the new dimensions based on the scale factor
    new_width = int(width * scale)
    new_height = int(height * scale)

    dim = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image

def create_folder(folderName):
    """
    Create a folder if it doesn't exist.

    Parameters
    ----------
    folderName : str
        The name of the folder to create.

    Returns
    -------
    None.

    """
    try:
        os.makedirs(folderName)
        print(f"The {folderName} directory is created!")
    except FileExistsError:
        # Directory already exists
        pass

    
def sample_gtsrb(train_dataset, num_samples_per_class):
    """
    This method takes a subset from a given GTSRB dataset.

    Parameters
    ----------
    train_dataset : torchvision.datasets.ImageFolder
        The train dataset.
    num_samples_per_class : int
        Number of images per class.

    Returns
    -------
    torchvision.datasets.Subset
        The subset dataset.

    """
    # Create a list to store the subset indices
    subset_indices = []

    # Create a list to store the true labels
    true_labels = []

    # Randomly sample the desired number of images from each class
    class_indices = train_dataset.class_to_idx

    for class_index in class_indices.values():
        class_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == class_index]

        subset_indices.extend(class_indices[:num_samples_per_class])

        # Store the true labels for the selected indices
        true_labels.extend([label for _, label in train_dataset.samples if label == class_index][:num_samples_per_class])

    # Create a subset dataset with the selected indices
    train_subset = Subset(train_dataset, subset_indices)

    # Assign the target feature to the train_subset dataset
    train_subset.targets = true_labels

    # Assign the classes to the train_subset dataset
    train_subset.classes = train_dataset.classes.copy()

    # Assign the images to the train_subset dataset
    img_list = [data for data, _ in train_subset]
    train_subset.images = img_list

    return train_subset

def visualize_data_distribution(dataset, dataset_classes, titel='Number of Images per Class'):
    """
    This method visualizes the distribution of a given dataset.

    Parameters
    ----------
    train_dataset : torchvision.datasets.ImageFolder
        The dataset to visualize.

    Returns
    -------
    None.

    """
    # Get the class labels and corresponding counts from the dataset
    class_labels = dataset.targets
    class_counts = {}
    for label in class_labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # Extract the class names from the dataset
    y = OrderedDict(sorted(class_counts.items()))

    def filter_dictionary(dictionary, keys):
        filtered_dict = {key: value for key, value in dictionary.items() if key in keys}
        return filtered_dict
    
    class_names = filter_dictionary(dataset_classes, y.keys())
    
    # Create a bar plot to visualize the number of images per class
    plt.figure(figsize=(16, 8))
    plt.bar(class_names.values(), y.values())
    plt.xlabel('Classess')
    plt.ylabel('Number of Images')
    plt.title(titel, fontsize=15)
    plt.xticks(rotation=90)

    # Add annotations to the bars
    for i, count in enumerate(y.values()):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.show()
    
