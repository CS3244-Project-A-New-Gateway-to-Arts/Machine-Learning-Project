#!/usr/bin/python3
import os
import shutil
import random

FROM_URL = '../data/resized/'
TO_URL = 'input/'
TRAIN_RATIO = 0.7

# filter into 11 artists with >200 images 
filtered = ['Vincent_van_Gogh', 'Edgar_Degas', 'Pablo_Picasso', 'Pierre-Auguste_Renoir', 'Albrecht_Dürer', 'Paul_Gauguin',
            'Francisco_Goya', 'Rembrandt', 'Alfred_Sisley', 'Titian', 'Marc_Chagall']

# Convert images into split folder input 
p = os.path.sep.join([FROM_URL])
imagePaths = os.listdir(p)

# Create a new dir for each class label
root = os.getcwd() + "/"
to_path = root + TO_URL
print(root)

def mkdir_if_not_exist(*paths):
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

train_dir, test_dir = os.path.join(to_path, 'train'), os.path.join(to_path, 'test')
mkdir_if_not_exist(train_dir, test_dir)

for i in imagePaths:
    label = "_".join(i.split(os.path.sep)[-1].split('_')[0:-1])
    if label not in filtered: 
        print("skip " + label)
        continue
    dirName = os.path.join(to_path, label)
    train_sub_dir = os.path.join(train_dir, label)
    test_sub_dir = os.path.join(test_dir, label)
    mkdir_if_not_exist(dirName, train_sub_dir, test_sub_dir) 

    # copy image into new dir
    rootImage = FROM_URL + i
    newPath = dirName + "/" + i.split(os.path.sep)[-1]
    shutil.copyfile(rootImage, newPath)

def train_test_split(lst, train_size):
    random.shuffle(lst)
    bound = int(train_size*len(lst))
    return (lst[:bound], lst[bound:])

# create train-test split
# for each folder
# get images names in a list
for folder in os.listdir(to_path):
    curr_folder_path = os.path.join(to_path, folder)
    images = os.listdir(curr_folder_path)
    train_img, test_img = train_test_split(images, train_size=TRAIN_RATIO)
    label = folder
    train_sub_dir = os.path.join(train_dir, label)
    test_sub_dir = os.path.join(test_dir, label)
    for img in train_img:
        img_src = os.path.join(curr_folder_path, img)
        print(img_src)
        shutil.move(img_src, train_sub_dir)
    for img in test_img:
        img_src = os.path.join(curr_folder_path, img)
        shutil.move(img_src, test_sub_dir)
