from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as nnF

import torchvision.datasets as datasets
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor

SCALE = 32
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

def gaussian_noise(image, noise_std=0.1, noise_mean=0.0):
    """
    Adds Gaussian noise to the input image.  
    """
    image = to_tensor(image)
    noise = torch.randn_like(image) * noise_std + noise_mean # calculate noise
    noisy_image = image + noise
    return noisy_image

def random_adjust_gamma(img):
    """
    Randomly adjusts the gamma value of the input image.  
    """
    gamma = 0.4 + 0.8  # Generate a random gamma value between 0.8 and 1.2
    return transforms.functional.adjust_gamma(img, gamma.item())

def rotate_by_angle(img, angle):
    """
    Rotates the input image by the specified angle.
    """
    return transforms.functional.rotate(img, angle)
    

class CustomTransformDataset(Dataset):
    """
    A custom dataset class that applies a given transform to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    transform : callable
        The transform function to be applied to selected samples.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.

    Attributes
    ----------
    dataset : Dataset
        The original dataset.
    desired_classes : list
        A list of desired classes.
    transform : callable
        The transform function to be applied to selected samples.
    filtered_data : list
        A list of filtered samples with labels in the desired_classes.
    targets : list
        A list of labels for the filtered samples.

    Methods
    -------
    __getitem__(self, index)
        Returns the transformed image and its label at the given index.
    __len__(self)
        Returns the length of the filtered dataset.
    filter_samples(self)
        Filters the original dataset to include only samples with labels in the desired_classes.

    """
    def __init__(self, dataset, transform, desired_classes=[]):
        self.dataset = dataset
        self.desired_classes = desired_classes
        self.transform = transform
        self.filtered_data, self.targets = self.filter_samples()

    def __getitem__(self, index):
        image, label = self.filtered_data[index]
        
        if self.transform: # transform image
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filtered_data)
    
    def filter_samples(self):
        """
        Filters the original dataset to include only samples with labels in the desired_classes.

        Returns
        -------
        list
            A list of filtered samples with labels in the desired_classes.
        list
            A list of labels for the filtered samples.

        """
        filtered = []
        targets = []
        for image, target in self.dataset:
            if target in self.desired_classes:
                filtered.append((image, target))
                targets.append(target)
        return filtered, targets

    
class CustomBlurTransform(CustomTransformDataset):
    """
    A custom dataset class that applies Gaussian blur and other transformations to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    resize_scale : int, optional
        The scale to resize the image before applying Gaussian blur and other transformations.
    radius : int, optional
        The radius of the Gaussian blur filter.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.
        
    """
    def __init__(self, dataset, resize_scale=SCALE, radius=3, desired_classes=[]):
        blur_transform = transforms.Compose([
            transforms.ToPILImage(), # convert to PIL image
            transforms.GaussianBlur(radius), # add Gaussian blur
            transforms.Resize((resize_scale, resize_scale)), # Resize
            transforms.ToTensor(), # convert to tensor
#             transforms.Normalize(mean=mean, std=std), # apply normalization if necessary
            
        ])
        super().__init__(dataset, transform=blur_transform, desired_classes=desired_classes)    

class CustomNoiseTransform(CustomTransformDataset):
    """
    A custom dataset class that adds Gaussian noise to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    resize_scale : int, optional
        The scale to resize the image before applying transformations.
    noise_std : float, optional
        The standard deviation of the Gaussian noise.
    noise_mean : float, optional
        The mean of the Gaussian noise.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.
    """

    def __init__(self, dataset, resize_scale=SCALE, noise_std=0.1, noise_mean=0.0, desired_classes=[]):
        noise_transform = transforms.Compose([
            transforms.ToPILImage(),            
            transforms.Lambda(lambda x: gaussian_noise(x, noise_std, noise_mean)), # add Gaussian noise
            transforms.ToPILImage(),
            transforms.Resize((resize_scale, resize_scale)),
            transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
        ])
        super().__init__(dataset, transform=noise_transform, desired_classes=desired_classes)

class CustomBrightnessTransform(CustomTransformDataset):
    """
    A custom dataset class that applies brightness transformation to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    resize_scale : int, optional
        The scale to resize the image before applying transformations.
    brightness_factor : tuple, optional
        The brightness factor range for the ColorJitter transformation.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.
    """
    
    def __init__(self, dataset, resize_scale=SCALE, brightness_factor=(0.3, 0.7), desired_classes=[]):
        brightness_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=brightness_factor), # brightness
            transforms.Resize((resize_scale, resize_scale)),
            transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
        ])
        super().__init__(dataset, transform=brightness_transform, desired_classes=desired_classes)
        
class CustomSaturationTransform(CustomTransformDataset):
    """
    A custom dataset class that applies saturation transformation to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    resize_scale : int, optional
        The scale to resize the image before applying transformations.
    saturation_factor : tuple, optional
        The saturation factor range for the ColorJitter transformation.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.
    """
    
    def __init__(self, dataset, resize_scale=SCALE, saturation_factor=(4,6), desired_classes=[]):
        saturation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(saturation=saturation_factor), # saturation
            transforms.Resize((resize_scale, resize_scale)),
            transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
        ])
        super().__init__(dataset, transform=saturation_transform, desired_classes=desired_classes)
        
class CustomContrastTransform(CustomTransformDataset):
    """
    A custom dataset class that applies contrast transformation to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    resize_scale : int, optional
        The scale to resize the image before applying transformations.
    contrast_factor : tuple, optional
        The contrast factor range for the ColorJitter transformation.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.
    """ 
    
    def __init__(self, dataset, resize_scale=SCALE, contrast_factor=(4,6), desired_classes=[]):
        contrast_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(contrast=contrast_factor), # contrast
            transforms.Resize((resize_scale, resize_scale)),
            transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
        ])
        super().__init__(dataset, transform=contrast_transform, desired_classes=desired_classes)
        

class CustomColorTransform(CustomTransformDataset):
    """
    A custom dataset class that applies color transformations (brightness, contrast, and saturation) to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    resize_scale : int, optional
        The scale to resize the image before applying transformations.
    brightness_factor : float, optional
        The brightness factor for the ColorJitter transformation.
    contrast_factor : tuple, optional
        The contrast factor range for the ColorJitter transformation.
    saturation_factor : tuple, optional
        The saturation factor range for the ColorJitter transformation.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.
    """
    
    def __init__(self, dataset, resize_scale=SCALE, 
                 brightness_factor=0.2, contrast_factor=(0.4, 0.6), saturation_factor=(0.4, 0.6), 
                 desired_classes=[]):
        color_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=brightness_factor, 
                                   contrast=contrast_factor, 
                                   saturation=saturation_factor), # color transformation
            transforms.Resize((resize_scale, resize_scale)),
            transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
        ])
        super().__init__(dataset, transform=color_transform, desired_classes=desired_classes)
        

class CustomRotationTransform(CustomTransformDataset):
    """
    A custom dataset class that applies random rotation to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    resize_scale : int, optional
        The scale to resize the image before applying transformations.
    rotation_factor : float, optional
        The range of random rotation angles in degrees.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.
    """
    
    def __init__(self, dataset, resize_scale=SCALE, rotation_factor=15, desired_classes=[]):
        rotation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(rotation_factor), # rotation
            transforms.Resize((resize_scale, resize_scale)),
            transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
        ])
        super().__init__(dataset, transform=rotation_transform, desired_classes=desired_classes)
        
class CustomResizeTransform(CustomTransformDataset):
    """
    A custom dataset class that resizes images to a specified scale to selected samples.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    resize_scale : int, optional
        The scale to resize the image before applying transformations.
    desired_classes : list, optional
        A list of desired classes. If provided, only samples with labels present in this list will be included.
    """
    
    def __init__(self, dataset, resize_scale=SCALE, desired_classes=[]):
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_scale, resize_scale)), # resize
            transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
        ])
        super().__init__(dataset, transform=resize_transform, desired_classes=desired_classes)
        
        
def apply_augmentation(dataset, scale, augmentations):
    """
    Applies a list of augmentations to the given dataset.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    scale : int
        The scale to resize the image before applying transformations.
    augmentations : list
        A list of tuples containing the augmentation class and its keyword arguments.

    Returns
    -------
    ConcatDataset
        The concatenated dataset with all applied augmentations.
    """
    
    transformed_datasets = []
    for aug_class, aug_kwargs in augmentations:
        transform_dataset = aug_class(dataset, resize_scale=scale, **aug_kwargs)
        transformed_datasets.append(transform_dataset)

    # Concatenate all transformed datasets
    full_dataset = ConcatDataset(transformed_datasets)

    # Get all labels of the concatenated datasets
    all_training_labels = []
    for dataset in full_dataset.datasets:
        all_training_labels.extend(dataset.targets)
    full_dataset.targets = all_training_labels

    return full_dataset
       