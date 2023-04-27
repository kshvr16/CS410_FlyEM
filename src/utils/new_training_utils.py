from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.metrics import MeanIoU


class NewTrainingUtils:
    @staticmethod
    def initialize_new_model(n_classes=2, activation_fn=('relu', 'sigmoid'), padding='same', optimizer_fn='adam', loss_fn='binary_crossentropy'):
        model = Sequential()
        # Encoder
        model.add(Conv2D(32, (3, 3), activation=activation_fn[0], padding='same', input_shape=(512, 512,1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation=activation_fn[0], padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation=activation_fn[0], padding='same'))
        # Decoder
        model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), activation=activation_fn[0], padding='same'))
        model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation=activation_fn[0], padding='same'))
        model.add(Conv2DTranspose(1, (3, 3), activation=activation_fn[1], padding='same'))
        return model

    @staticmethod
    def compile_new_model(model, n_classes=2, optimizer_fn='adam', loss_fn='binary_crossentropy'):
        IoU = MeanIoU(num_classes=n_classes)
        model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=[IoU]) 
        return model
