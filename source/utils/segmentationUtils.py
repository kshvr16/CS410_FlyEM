import os
import shutil
import numpy as np
import mahotas as mh
import imageio as im
from utils import Utils
from utils import ImageUtils
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from skimage import measure, filters


class SegmentationUtils:

    @staticmethod
    def segmentation_process(cropped, normalized, originals, masks):
        success_count = 0
        for z in tqdm(sorted(os.listdir(cropped))):
            if z.endswith("tif"):  # checking the file ends with tif
                img = im.imread(os.path.join(cropped, z))
                print(z)
                plt.imshow(img)
                # plt.show()

                # Apply a Gaussian filter to the image
                c = img.copy()
                # b = mh.gaussian_filter(b, sigma=3)

                # Set values below 100 to 0
                for a in range(150, 0, -1):
                    Utils.start_time()
                    b = img.copy()
                    b = mh.gaussian_filter(b, sigma=3)
                    b[b < a] = 0
                    # print (a)
                    # b = exposure.equalize_hist(b)
                    # Label the regions in the filtered image
                    labeled, number = mh.label(b)

                    # filter based on labeled region size
                    sizes = mh.labeled.labeled_size(labeled)

                    # Remove the regions that are less than 1000
                    too_small = np.where(sizes < 1500)
                    labeled_only_big = mh.labeled.remove_regions(labeled, too_small)

                    # too_large = np.where(sizes > 20500)
                    # labeled_only_big = mh.labeled.remove_regions(labeled, too_large)
                    # for debug
                    # plt.imshow(labeled_only_big)
                    # plt.show()

                    # Create a binary mask from the filtered labeled regions
                    binary_mask = labeled_only_big.copy()
                    binary_mask[binary_mask > 0] = 1
                    labeled, number_1 = mh.label(binary_mask)

                    # Close the regions in the binary mask
                    binary_mask_closed = mh.morph.close(binary_mask)

                    plt.figure(figsize=(10, 10))
                    # plt.imshow(binary_mask_closed)
                    # plt.show()

                    # Set a threshold for the minimum region size
                    min_region_size = 3000

                    # Initialize a variable to count the number of regions above the minimum size
                    large_regions = 0

                    # Get the sizes of the labeled regions
                    region_sizes = measure.regionprops(labeled, intensity_image=binary_mask_closed)

                    # Iterate over the region sizes and count the number of large regions
                    for region in region_sizes:
                        if region.area > min_region_size:
                            large_regions += 1

                    threshold = filters.threshold_otsu(binary_mask_closed)
                    binary_image = binary_mask_closed > threshold
                    print('time taken for iteration', a, 'image', z, 'is:')
                    Utils.end_time()
                    #             if number_1>= 90:
                    #                 print (z)
                    #                 plt.imshow(binary_image)
                    #                 plt.show()
                    #                 print (number_1)
                    #                 print (threshold)
                    #                 print(large_regions)

                    if 150 >= number_1 >= 100:
                        if large_regions <= 20:  # 20 is ideal value
                            print("######################################################################")
                            print(z)
                            # plt.figure(figsize=(10,10))
                            print("The image has clear segmentation.")
                            # plt.imshow(binary_image)
                            # plt.show()
                            print(number_1)
                            print(threshold)
                            print(large_regions)
                            shutil.move(os.path.join(normalized, z), os.path.join(originals, z))
                            shutil.move(os.path.join(cropped, z), os.path.join(masks, z))
                            mh.imsave(os.path.join(masks, z), binary_image)
                            # print (sizes)
                            print("######################################################################")
                            success_count += 1
                            print(success_count)
                            break
        print(success_count)
        print('######################################### DONE        ############################################')

    @staticmethod
    def segment_cpu(src_dir, dest_dir):
        real = f"{dest_dir}/real_images"
        directories = {
            "normalized": f"{dest_dir}/normalized_images",
            "cropped": f"{dest_dir}/cropped_images",
            "npy": f"{dest_dir}/pre_processing/npy",
            "originals": f"{dest_dir}/pre_processing/originals",
            "masks": f"{dest_dir}/pre_processing/masks",
            "test": f"{dest_dir}/pre_processing/test",
            "s": f"{dest_dir}/S",
            "crop_original": f"{dest_dir}/pre_processing/crop_originals",
            "crop_masks": f"{dest_dir}/pre_processing/crop_masks"
        }

        # Check and creation of all the directories
        Utils.check(real)
        for key in directories:
            Utils.check(directories[key])

        # Copying of original images to the required directories
        Utils.a2b_copy(src_dir, real)
        Utils.count_files(directories["originals"])  # count of the image files copied to the originals directory
        Utils.a2b_copy(directories, directories["normalized"])
        Utils.a2b_copy(directories, directories["cropped"])

        # Normalization to the original images
        ImageUtils.norm(directories["normalized"])
        ImageUtils.norm(directories["cropped"])

        # Cropping of the normalized images
        ImageUtils.crop(directories["cropped"])

        # Segmentation process
        SegmentationUtils.segmentation_process(cropped=directories["cropped"],
                                               normalized=directories["normalized"],
                                               originals=directories["originals"],
                                               masks=directories["crop_masks"])
