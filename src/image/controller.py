from image.model import Image
from image.view import ImageView


class ImageController:

    @staticmethod
    def read(filepath):
        return Image.read(filepath)

    @staticmethod
    def normalize(data):
        return Image.normalize(data)

    @staticmethod
    def center_crop(data, crop_size=(512, 512)):
        return Image.center_crop(data, crop_size)

    @staticmethod
    def smooth(data, sigma_value=3):
        return Image.smooth(data, sigma_value)

    @staticmethod
    def threshold(data, threshold_value):
        return Image.threshold(data, threshold_value)

    @staticmethod
    def label(data):
        return Image.label(data)

    @staticmethod
    def select_regions(data, region_size):
        return Image.select_regions(data, region_size)

    @staticmethod
    def binary_mask(data):
        return Image.binary_mask(data)

    @staticmethod
    def close_binary_mask(data):
        return Image.close_binary_mask(data)

    @staticmethod
    def binary_image(data):
        return Image.binary_image(data)

    @staticmethod
    def display(data):
        ImageView.display(data)

    @staticmethod
    def classify_image(data):
        return Image.classify_image(data)
