import os
import shutil
import random

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_data(source, train, val, test, train_ratio=0.7, val_ratio=0.2):
    for root, dirs, files in os.walk(source):
        for cls in dirs:
            cls_path = os.path.join(root, cls)
            images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
            random.shuffle(images)
            train_split = int(train_ratio * len(images))
            val_split = int(val_ratio * len(images)) + train_split

            train_images = images[:train_split]
            val_images = images[train_split:val_split]
            test_images = images[val_split:]

            for img in train_images:
                src_path = os.path.join(cls_path, img)
                dest_path = os.path.join(train, os.path.relpath(cls_path, source))
                create_dir(dest_path)
                shutil.copy(src_path, dest_path)

            for img in val_images:
                src_path = os.path.join(cls_path, img)
                dest_path = os.path.join(val, os.path.relpath(cls_path, source))
                create_dir(dest_path)
                shutil.copy(src_path, dest_path)

            for img in test_images:
                src_path = os.path.join(cls_path, img)
                dest_path = os.path.join(test, os.path.relpath(cls_path, source))
                create_dir(dest_path)
                shutil.copy(src_path, dest_path)

# Example usage
source_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\labeled-images"
train_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\input_split\train"
val_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\input_split\valid"
test_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\input_split\test"

split_data(source_dir, train_dir, val_dir, test_dir)
