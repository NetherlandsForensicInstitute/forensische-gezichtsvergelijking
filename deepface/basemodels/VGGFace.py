import os
from pathlib import Path

import gdown
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, \
    Flatten, \
    Dropout, Activation, Input
from tensorflow.keras.models import Model, Sequential


def baseModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    return model


def loadModel():
    model = baseModel()
    home = str(Path.home())
    weights_path = home + '/.deepface/weights/vgg_face_weights.h5'
    _maybe_download_weights(weights_path)
    model.load_weights(weights_path)

    # We take layer `-5` as output because it's the final embedding layer. The
    # remaining layers are meant for classifying the image as one of the 2622
    # identities, which is not what we're interested in.
    return Model(inputs=model.layers[0].input,
                 outputs=model.layers[-5].output)


def load_training_model():
    base_model = loadModel()
    batch_shape, *input_shape = base_model.input_shape

    anchor_input = Input(input_shape, batch_shape)
    positive_input = Input(input_shape, batch_shape)
    negative_input = Input(input_shape, batch_shape)

    anchor_output = base_model(anchor_input)
    positive_output = base_model(positive_input)
    negative_output = base_model(negative_input)

    output = tf.stack([
        tf.squeeze(anchor_output, axis=[1, 2]),
        tf.squeeze(positive_output, axis=[1, 2]),
        tf.squeeze(negative_output, axis=[1, 2]),
    ], axis=1)

    return Model(inputs=[anchor_input, positive_input, negative_input],
                 outputs=output)


def _maybe_download_weights(path):
    """
    Checks if the VGGFace weights exist, and if not, downloads them.

    :param path: str, the path to where the weights should be stored
    """
    if not os.path.exists(path):
        print("vgg_face_weights.h5 will be downloaded...")
        url = 'https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo'
        gdown.download(url, path, quiet=False)
