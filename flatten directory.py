import os
import shutil

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def flatten_data(source_dir, flat_structure):
    # Traverse the train, val, and test directories
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(source_dir, split)
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # Extract the class directory name
                    relative_path = os.path.relpath(root, split_dir)
                    class_name = os.path.basename(relative_path)
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(flat_structure, split, class_name, file)

                    # Create the class directory in the flat structure if it doesn't exist
                    create_dir(os.path.join(flat_structure, split, class_name))
                    # Move or copy the image to the new flat structure
                    shutil.move(src_path, dest_path)  # Use shutil.copy if you want to copy instead of move

# Paths
source_dir = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\input_split"
flat_structure = r"D:\Hyper-Kvasir Dataset\hyper-kvasir-labeled-images\flat_structure"

# Flatten the data
flatten_data(source_dir, flat_structure)
