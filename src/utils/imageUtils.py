import os
import time
import shutil

import matplotlib.pyplot as plt
import numpy as np
import mahotas as mh
from skimage import measure, filters


class ImageUtils:
    @staticmethod
    def read_image(path):
        return mh.imread(path)

    @staticmethod
    def crop_image(original_image, crop_size):
        # returns cropped image of crop size at the center of the image
        height, width = original_image.shape
        center_x, center_y = height//2, width//2
        new_height, new_width = crop_size
        start_x, start_y = center_x - (new_height//2), center_y - (new_width//2)
        end_x, end_y = center_x + (new_height//2), center_y + (new_width//2)
        result_image = original_image[start_x:end_x, start_y:end_y]
        return result_image

    @staticmethod
    def normalize_image(original_image):
        # returns normalized image to pixel range of 0 to 255 from the original pixel range
        result_image = original_image.astype(np.float64)
        result_image /= result_image.max()
        return result_image * 255

    @staticmethod
    def get_multiple_crops(original_image, crop_size):
        # returns list of all the cropped images of given crop size
        cropped_images = []
        height, width = original_image.shape
        new_height, new_width = crop_size.shape
        for i in range(0, height, new_height):
            for j in range(0, width, new_width):
                cropped_images.append(original_image[i:i + new_height, j:j + new_width])
        return cropped_images

    @staticmethod
    def get_overlay(original_image, mask_image):
        # returns an overlay image from the original image and mask image
        result_image = np.copy(original_image)
        np.putmask(result_image, mask_image.astype(bool), 0)
        return result_image

    @staticmethod
    def apply_gaussian_filter(original_image, sigma_value):
        # returns the gaussian filtered image of the original image
        result_image = mh.gaussian_filter(original_image, sigma=sigma_value)
        return result_image

    @staticmethod
    def apply_threshold(original_image, threshold_value):
        # returns the threshold image of the original image for the given threshold value
        result_image = original_image.copy()
        result_image[result_image < threshold_value] = 0
        return result_image

    @staticmethod
    def apply_region_labelling(original_image):
        # returns the labeled regions and number of regions
        labeled_result, nr_objects_result = mh.label(original_image)
        return labeled_result, nr_objects_result

    @staticmethod
    def remove_small_regions(labeled, region_size):
        # return the resultant image after removing small regions
        sizes = mh.labeled.labeled_size(labeled)
        too_small = np.where(sizes < region_size)
        labeled_result = mh.labeled.remove_regions(labeled, too_small)
        return labeled_result

    @staticmethod
    def get_binary_mask(labeled):
        # returns a binary mask of the input image
        result_mask = labeled.copy()
        result_mask[result_mask > 0] = 1
        return result_mask

    @staticmethod
    def get_closed_binary_mask(binary_mask):
        # return the closed binary mask of the given binary mask
        result_mask = mh.morph.close(binary_mask)
        return result_mask

    @staticmethod
    def get_binary_image(binary_mask):
        # returns the binary image for the given binary mask
        # threshold_value = mh.otsu(binary_mask)
        # result_image = binary_mask > threshold_value
        # return result_image
        threshold = filters.threshold_otsu(binary_mask)
        result_binary_image = binary_mask > threshold
        return result_binary_image

    @staticmethod
    def verify_image_segmentation(binary_mask):
        # verifies the segmentation of the image is clear or not
        binary_mask_closed = ImageUtils.get_closed_binary_mask(binary_mask)
        labeled_binary, nr_objects_binary = ImageUtils.apply_region_labelling(binary_mask)
        region_sizes = measure.regionprops(labeled_binary, intensity_image=binary_mask_closed)
        large_regions = 0
        min_region_size = 3000
        for region in region_sizes:
            if region.area > min_region_size:
                large_regions += 1
        if 100 <= nr_objects_binary <= 150 and large_regions <= 20:
            return True
        else:
            return False

    @staticmethod
    def get_start_time():
        # returns start time of the execution
        start_time = time.time()
        return start_time

    @staticmethod
    def get_execution_duration(start_time):
        # returns the total duration of the execution
        end_time = time.time()
        return end_time - start_time

    @staticmethod
    def create_new_dir(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        else:
            os.mkdir(dir_path)

    @staticmethod
    def convert_to_npy(image_data):
        # returns the numpy data of image data
        numpy_data = np.array(image_data)
        return numpy_data

    @staticmethod
    def src2dest_copy(src_dir, dest_dir, file):
        # copies the specified file from source dir to destination dir
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

    @staticmethod
    def display_image(image_data):
        plt.imshow(image_data)
        plt.show()

    # import random
    # from tqdm import tqdm
    # def src2dest_copy(src_dir, dest_dir):
    #     # copies all the ".tif" files from source dir to destination dir
    #     for file in tqdm(sorted(os.listdir(src_dir))):
    #         if file.endswith("tif"):
    #             shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))
    #
    #
    # def src2dest_random_copy(src_dir, dest_dir, num_of_images):
    #     # copies the given number of random images from source dir to destination dir
    #     files = [file for file in tqdm(os.listdir(src_dir)) if file.endswith(".tif")]
    #     selected_files = random.sample(files, num_of_images)
    #     for file in tqdm(selected_files):
    #         shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))


    # def segmentation_pipeline(original_image):
    #     normalized_image = imageUtils.get_normalized_image(original_image)
    #     cropped_image = imageUtils.get_cropped_image(normalized_image)
    #     filtered_image = imageUtils.apply_gaussian_filter(cropped_image)
    #     threshold_image = imageUtils.apply_threshold(filtered_image, threshold_value=55)
    #     labeled, nr_objects = imageUtils.apply_region_labelling(threshold_image)
    #     labeled_big = imageUtils.remove_small_regions(labeled, region_size=1500)
    #     binary_mask = imageUtils.get_binary_mask(labeled_big)
    #     binary_mask_closed = imageUtils.get_closed_binary_mask(binary_mask)
    #     binary_image = imageUtils.get_binary_image(binary_mask_closed)

    #     if imageUtils.verify_image_segmentation(binary_mask):
    #         print("The image has a clear segmentation")
    #     else:
    #         print("The image doesn't have a clear segmentation")
    #     return binary_image
