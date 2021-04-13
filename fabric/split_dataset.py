from shutil import copyfile
import random
import os
import splitfolders

base_folder = '/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/dataset/'
output_dir = '/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/output_dataset'
# splitfolders.ratio(base_folder,output_dir,ratio=(0.7,0.2,0.1),group_prefix=None)
validation = '/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/validation'

# Copyright 2014-2017 Bert Carremans
# Author: Bert Carremans <bertcarremans.be>
#
# License: BSD 3 clause


def img_train_test_split(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure
    
    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path   
        
    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    train_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/val"
    val_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/for_inference"
    train_counter = 0
    validation_counter = 0

    # Randomly assign an image to train or validation folder
    for filename in os.listdir(img_source_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # fileparts = filename.split('.')

            if random.uniform(0, 1) <= train_size:
                copyfile(os.path.join(img_source_dir, filename), os.path.join(train_dir,filename))
                train_counter += 1
            else:
                copyfile(os.path.join(img_source_dir, filename), os.path.join(val_dir,filename))
                validation_counter += 1


img_train_test_split(validation,0.75)
