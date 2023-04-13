import keras
import numpy as np
import keras.optimizers
from keras_unet.utils import plot_imgs
from keras_unet.models import custom_unet
from keras_unet.utils import plot_segm_history
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou, iou_thresholded


class TrainingUtils:

    @staticmethod
    def load_file(filepath):
        # returns a numpy array of pixel values
        return np.load(filepath)

    @staticmethod
    def resize_image(data):
        # returns a re-sized array of shape (512, 512, 1)
        return data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
    
    @staticmethod
    def random_permutation(length):
        # creates and returns an array of numbers in random order of given specific length
        return np.random.permutation(length)

    @staticmethod
    def randomize_data(data, idx_array):
        # returns an randomized data based on the pixels organized using random_permutation()
        return data[idx_array]

    @staticmethod
    def individual_normalize(data):
        # returns an normalized data for each individual image
        result_data = data.astype(np.float64)
        for idx in range(result_data.shape[0]):
            result_data[idx] = (result_data[idx] - result_data[idx].min()) / (result_data[idx].max() - result_data[idx].min())
        return result_data

    @staticmethod
    def train_val_test_split(data, train_part, val_part, test_part):
        # returns a split array of training, validation and testing
        length = len(data)
        train_end = int(train_part * length)
        val_end = int((train_part + val_part) * length)
        return data[: train_end], data[train_end+1: val_end], data[val_end+1:]

    @staticmethod
    def init_model(n_filters=32, dropout_val=0.5, activation_function="sigmoid"):
        # returns an UNet Neural networks with the given specifications
        return custom_unet(input_shape=(512, 512, 1), use_batch_norm=False, num_classes=1,
                           filters=n_filters, dropout=dropout_val, output_activation=activation_function)

    @staticmethod
    def compile_model(model, optimizer_fn="Adam", loss_fn="binary_crossentropy", metrics_arr=(iou, iou_thresholded)):
        # compiles the model with the given optimizzer and loss fucntion and with given metrics
        model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metrics_arr)
        return model

    @staticmethod
    def training(model, X_train, y_train, X_val, y_val, batch, epochs):
        # returns the history of training the model by using fit()
        print("training and prediction for", "batch size:  ", batch, "epochs:  ", epochs)
        history = model.fit(X_train, y_train, batch, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
        return history

    @staticmethod
    def plot_predict(history, model, X_test, y_test):
        # plots the metrics over time and by evaluating the trained model with testing data
        plot_segm_history(history)
        y_pred = model.predict(X_test)
        plot_imgs(org_imgs=X_test, mask_imgs=y_test, pred_imgs=y_pred, nm_img_to_plot=10)
