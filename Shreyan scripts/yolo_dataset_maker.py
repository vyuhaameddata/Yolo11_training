# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 21:11:41 2023

@author: Lucid

edited on 24 feb 2025 by shreyan
added merge function add to existing dataset 

documented on 21 feb 2025 by Shreyan
function: given a folder with images and labels sub folders it generates training data suitable for yolo dividng it in Train,Val,Test
"""
import random

import os
import shutil
from sklearn.model_selection import train_test_split

def move_files_to_folder(list_of_files, destination_folder):
        for f in list_of_files:
            try:
                shutil.move(f, destination_folder)
            except:
                print(f)
                assert False
    
def create_cell_dataset(folder, dir_):
    images_folder = os.path.join(folder,'images')
    labels_folder = os.path.join(folder,'labels')
    images = [os.path.join(images_folder, x) for x in os.listdir(images_folder) if x[-3:] == "tif"]
    annotations = [os.path.join(labels_folder, x) for x in os.listdir(labels_folder) if x[-3:] == "txt"]
    
    images.sort()
    annotations.sort()
    print(len(images), len(annotations))
    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    #val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
    
    train_folder = os.path.join(dir_,'train')
    val_folder = os.path.join(dir_,'val')
    #test_folder = os.path.join(folder,'test')
    
    
    if not os.path.isdir(train_folder) : os.mkdir(train_folder)
    if not os.path.isdir(val_folder) : os.mkdir(val_folder)
    #if not os.path.isdir(test_folder) : os.mkdir(test_folder)
    
    
    train_images_folder = os.path.join(train_folder,'images')
    train_labels_folder = os.path.join(train_folder,'labels')
    val_images_folder = os.path.join(val_folder,'images')
    val_labels_folder = os. path.join(val_folder,'labels')
    
    if not os.path.isdir(train_images_folder) : os.mkdir(train_images_folder)
    if not os.path.isdir(val_images_folder) : os.mkdir(val_images_folder)
  
    if not os.path.isdir(train_labels_folder) : os.mkdir(train_labels_folder)
    if not os.path.isdir(val_labels_folder) : os.mkdir(val_labels_folder)
    
    #Utility function to move images 
    

    # Move the splits into their folders
    move_files_to_folder(train_images,train_images_folder)
    move_files_to_folder(val_images, val_images_folder)
  
    move_files_to_folder(train_annotations, train_labels_folder)
    move_files_to_folder(val_annotations,val_labels_folder)
  

def create_yolo_dataset(folder):
    images_folder = os.path.join(folder,'images')
    labels_folder = os.path.join(folder,'labels')
    
    images = [os.path.join(images_folder, x) for x in os.listdir(images_folder) if x[-3:] == "tif"]
    annotations = [os.path.join(labels_folder, x) for x in os.listdir(labels_folder) if x[-3:] == "txt"]
    
    images.sort()
    annotations.sort()
    
    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.3, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
    
    train_folder = os.path.join(folder,'train')
    val_folder = os.path.join(folder,'val')
    test_folder = os.path.join(folder,'test')
    
    if not os.path.isdir(train_folder) : os.mkdir(train_folder)
    if not os.path.isdir(val_folder) : os.mkdir(val_folder)
    if not os.path.isdir(test_folder) : os.mkdir(test_folder)
    
    
    train_images_folder = os.path.join(train_folder,'images')
    train_labels_folder = os.path.join(train_folder,'labels')
    val_images_folder = os.path.join(val_folder,'images')
    val_labels_folder = os. path.join(val_folder,'labels')
    test_images_folder = os.path.join(test_folder,'images')
    test_labels_folder = os.path.join(test_folder,'labels')
    
    if not os.path.isdir(train_images_folder) : os.mkdir(train_images_folder)
    if not os.path.isdir(val_images_folder) : os.mkdir(val_images_folder)
    if not os.path.isdir(test_images_folder) : os.mkdir(test_images_folder)
    if not os.path.isdir(train_labels_folder) : os.mkdir(train_labels_folder)
    if not os.path.isdir(val_labels_folder) : os.mkdir(val_labels_folder)
    if not os.path.isdir(test_labels_folder): os.mkdir(test_labels_folder)
           
    
    #Utility function to move images 

    # Move the splits into their folders

    move_files_to_folder(train_images,train_images_folder)
    move_files_to_folder(val_images, val_images_folder)
    move_files_to_folder(test_images,test_images_folder)
    move_files_to_folder(train_annotations, train_labels_folder)
    move_files_to_folder(val_annotations,val_labels_folder)
    move_files_to_folder(test_annotations, test_labels_folder)

def merge_yolo_dataset(folder, des):
    images_folder = os.path.join(folder,'images')
    labels_folder = os.path.join(folder,'labels')
    
    images = [os.path.join(images_folder, x) for x in os.listdir(images_folder) if x[-3:] == "tif"]
    annotations = [os.path.join(labels_folder, x) for x in os.listdir(labels_folder) if x[-3:] == "txt"]
    
    images.sort()
    annotations.sort()
    
    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.3, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
    
    train_folder = os.path.join(des,'train')
    val_folder = os.path.join(des,'val')
    test_folder = os.path.join(des,'test')
    
    if not os.path.isdir(train_folder) : os.mkdir(train_folder)
    if not os.path.isdir(val_folder) : os.mkdir(val_folder)
    if not os.path.isdir(test_folder) : os.mkdir(test_folder)
    
    
    train_images_folder = os.path.join(train_folder,'images')
    train_labels_folder = os.path.join(train_folder,'labels')
    val_images_folder = os.path.join(val_folder,'images')
    val_labels_folder = os. path.join(val_folder,'labels')
    test_images_folder = os.path.join(test_folder,'images')
    test_labels_folder = os.path.join(test_folder,'labels')
    
    if not os.path.isdir(train_images_folder) : os.mkdir(train_images_folder)
    if not os.path.isdir(val_images_folder) : os.mkdir(val_images_folder)
    if not os.path.isdir(test_images_folder) : os.mkdir(test_images_folder)
    if not os.path.isdir(train_labels_folder) : os.mkdir(train_labels_folder)
    if not os.path.isdir(val_labels_folder) : os.mkdir(val_labels_folder)
    if not os.path.isdir(test_labels_folder): os.mkdir(test_labels_folder)

        
    #Utility function to move images 

    # Move the splits into their folders

    move_files_to_folder(train_images,train_images_folder)
    move_files_to_folder(val_images, val_images_folder)
    move_files_to_folder(test_images,test_images_folder)
    move_files_to_folder(train_annotations, train_labels_folder)
    move_files_to_folder(val_annotations,val_labels_folder)
    move_files_to_folder(test_annotations, test_labels_folder)