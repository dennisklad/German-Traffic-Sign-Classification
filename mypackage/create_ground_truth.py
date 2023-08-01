"""
Create ground truth CSV files for all classes and each class separately.

"""

import cv2
import os

# The path to the directory containing the training data
PATH = r'/mnt/c/Users/Nosni/Local Documents/Github/DCAITI_Project/FinalDataset/Final_Training'

HEADER = 'filename,heigth,width,label\n'

# This is the file containing the ground truth for all classes!
with open(f'{PATH}/ground-truth.csv', 'w') as f: 
    
    # Write the header to the file
    f.write(HEADER)
    
    # For each class folder in the PATH  (00000 -  000049)
    for folder_index, folder_name in enumerate(os.listdir(PATH)):
        
        if '.' in folder_name:  # In case this is not a folder!
            continue
            
        print(f'Folder {folder_index:2}: {folder_name}')
        
        # This is the class ground truth!
        with open(f'{PATH}/{folder_name}/ground_truth_{folder_name}.csv', 'w') as f_class:

            # Write the header to the file
            f_class.write(HEADER)
            
            # For all images in the specific class folder
            for image_index, image_name in enumerate(os.listdir(f'{PATH}/{folder_name}')):

                temp_image = cv2.imread(f'{PATH}/{folder_name}/{image_name}')

                if temp_image is None:  # In case this is not an image!
                    continue
                    
                height, width = temp_image.shape[:2]

                # Write the ground truth to both files
                f.write(f'{image_name},{height},{width},{folder_index}\n')
                f_class.write(f'{image_name},{height},{width},{folder_index}\n')

        print(f'{image_index+1:5} images were read in folder {folder_name}\n')