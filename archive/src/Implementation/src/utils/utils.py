import os
import time
import shutil
import random
import inspect
import numpy as np
import mahotas as mh
from PIL import Image
from tabulate import tabulate
from tqdm.notebook import tqdm

start_time_var = 0  # Define start_time in the global scope


class Utils:

    @staticmethod
    def start_time():
        global start_time_var  # Use the global keyword to access the global start_time variable
        start_time_var = time.time()
        # hint: start_time() - To start timer.

    @staticmethod
    def end_time():
        end_time_var = time.time()
        execution_time = end_time_var - start_time_var
        print(f"Execution time: {execution_time:.2f} seconds")
        # hint: end_time() - To end timer

    @staticmethod
    def check(path):
        # Create the directory if it does not exist
        os.makedirs(path, exist_ok=True)

        # Remove the directory and all its contents
        shutil.rmtree(path)

        # Create a new empty directory
        os.mkdir(path)
        # hint: check(path) - To recreate a particular path

    @staticmethod
    def npy_conversion(tif_dir, npy_path):
        tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
        data = []
        for tif_file in tqdm(tif_files):
            img = Image.open(os.path.join(tif_dir, tif_file))
            data.append(np.array(img))
        np.save(npy_path, data)
        # hint: npy_conversion(path , npy + '/filename.npy' ) -To create NPY files

    @staticmethod
    def a2b_copy(path1, path2):
        for z in tqdm(sorted(os.listdir(path1))):
            if z.endswith("tif"):
                shutil.copy(os.path.join(path1, z), os.path.join(path2, z))
                # hint: a2b_copy(src_path, dest_path) - To copy all images

    @staticmethod
    def a2b_random(src_dir, dst_dir, number):
        # Get a list of all image files in the source directory
        image_files = [f for f in tqdm(os.listdir(src_dir)) if f.endswith('.tif')]
        # Randomly select given images from the list
        selected_images = random.sample(image_files, number)
        # Copy the selected images to the destination directory
        for image in selected_images:
            src_path = os.path.join(src_dir, image)
            dst_path = os.path.join(dst_dir, image)
            shutil.copy2(src_path, dst_path)
        # Print a message when done
        print('Copied 10 random images to', dst_dir)
        # hint: a2b_random(src_path, dest_path, random_number) - To copy n random images

    @staticmethod
    def count_files(dir_path):
        if os.path.isdir(dir_path):
            file_count = 0
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                file_size_bytes = os.path.getsize(file_path)
                file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
                # print(f'{file_name} - Size: {file_size_mb} MB')
                file_count += 1
            return file_count
        else:
            print(f"{dir_path} is not a valid directory")
            return 0
            # hint: count_files(path) - To count number of files in that path

    @staticmethod
    def shape(raw):
        for z in tqdm(sorted(os.listdir(raw))):
            if z.endswith("tif"):
                img = mh.imread(os.path.join(raw, z))
                print(img.shape)
                # hint: shape(path) - To print shape of all the images in the path

    @staticmethod
    def refresh(experiment: str, directories: dict):
        for key in directories:
            if os.path.exists(directories[key]):
                shutil.rmtree(directories[key])
            os.makedirs(directories[key])
            # hint: refresh("experiment name", directories) - to recreate all directories in that dict

    @staticmethod
    def paths(directories):
        for key, value in directories.items():
            globals()[key] = value
        return directories
        # hint: paths(directories) - To call the directories outside the dictionary

    @staticmethod
    def help():
        functions = [value for key, value in globals().items() if inspect.isfunction(value)]
        headers = ["Function", "Hint", "Used for"]
        data = []
        for func in functions:
            source = inspect.getsource(func)
            lines = source.split("\n")
            hint_line = [line for line in lines if line.strip().startswith("#hint:")]
            if hint_line:
                hint_parts = hint_line[0].split("#hint:")[1].strip().split(" - ")
                hint_text = hint_parts[0]
                usage_text = hint_parts[1] if len(hint_parts) > 1 else ""
                data.append([f"{func.__name__}()", hint_text, usage_text])
        print(tabulate(data, headers=headers))
