import os
import random
import numpy as np
import mahotas as mh
from PIL import Image
from utils import Utils
from tqdm.notebook import tqdm


class ImageUtils:

    @staticmethod
    def crop(test):
        for z in tqdm(sorted(os.listdir(test))):
            if z.endswith("tif"):  # checking the file ends with tif
                # Read in the image
                img = mh.imread(os.path.join(test, z))
                img_cropped = img[1000:2500, 2500:4500]
                mh.imsave(os.path.join(test, z), img_cropped)
                print(z)
                # hint: crop(spath) - To crop all images

    @staticmethod
    def norm(path):
        for z in tqdm(sorted(os.listdir(path))):
            # added interactive progressbar to decrease the uncertainty and to increase curiosity :)
            if z.endswith("tif"):  # checking the file ends with tif
                img = mh.imread(os.path.join(path, z))
                # Normalize the image
                img = img.astype(np.float64)
                img /= img.max()
                img *= 255
                # Save the processed image back to the temporary directory
                mh.imsave(os.path.join(path, z), img)
                # hint: norm(path) - To normalize all the images in the path

    @staticmethod
    def heavycrop(test):
        Utils.start_time()
        for z in tqdm(sorted(os.listdir(test))):
            if z.endswith("tif"):
                # Read in the image
                img = mh.imread(os.path.join(test, z))

                # Calculate the number of crops in each dimension
                height, width = img.shape[:2]
                num_crops_y = height // 512
                num_crops_x = width // 512

                for i in range(num_crops_y):
                    for j in range(num_crops_x):
                        # Crop the image
                        start_y = i * 512
                        start_x = j * 512
                        img_cropped = img[start_y:start_y + 512, start_x:start_x + 512]

                        # Create a new file name for the cropped image
                        file_name, file_ext = os.path.splitext(z)
                        new_file_name = f"{file_name}_{i}_{j}{file_ext}"

                        # Save only if cropped image has shape (512, 512)
                        if img_cropped.shape == (512, 512):
                            mh.imsave(os.path.join(test, new_file_name), img_cropped)
                        else:
                            print(f"Warning: Cropped image has unexpected shape {img_cropped.shape}")

                Utils.end_time()
                # Remove original image file after cropping is done.
                os.remove(os.path.join(test, z))
                # hint: heavy_crop(spath) - To heavy crop all images 512x512

    @staticmethod
    def overlay(original_dir, mask_dir, n):
        original_images = os.listdir(original_dir)
        mask_images = os.listdir(mask_dir)

        for i in range(n):
            original_image = random.choice(original_images)
            mask_image = original_image.replace('.png', '_mask.png')
            if mask_image in mask_images:
                img = Image.open(os.path.join(original_dir, original_image))
                mask = Image.open(os.path.join(mask_dir, mask_image))
                img.paste(mask, (0, 0), mask)
                img.show()
                # hint: # overlay_masks('original', 'masks', 3) - To plot overlay wiht masks
