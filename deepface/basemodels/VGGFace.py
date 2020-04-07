import os
from pathlib import Path

import gdown
import tensorflow as tf
from tensorflow.keras.layers import (Convolution2D,
                                     ZeroPadding2D,
                                     MaxPooling2D,
                                     Flatten,
                                     Dropout,
                                     Activation,
                                     Input)
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

    # We take layer -5 as output because it's the final embedding layer. The
    # remaining layers are meant for classifying the image as one of the 2622
    # identities, which is not what we're interested in. We flatten the output
    # of this convolutional embedding layer by squeezing the two spatial
    # dimensions (axis=[1, 2]) so that the resulting embedding tensors have
    # shape `(batch_size, embedding_size)`.
    embeddings = tf.squeeze(model.layers[-5].output, axis=[1, 2])

    # Ensure that the embeddings are l2 normalized for stability during
    # training.
    embeddings = tf.math.l2_normalize(embeddings, axis=1)
    return Model(inputs=model.layers[0].input, outputs=embeddings)


def load_training_model() -> tf.keras.Model:
    """
    Returns a tf.keras.Model instance that can be used to finetune the learned
    embeddings. This training model takes 3 inputs, namely:

        anchor: A 4D tensor containing a batch of anchor images with shape
            `(batch_size, height, width, num_channels)`.
        positive: A 4D tensor containing a batch of images of the same identity
            as the anchor image with shape `(batch_size, height, width,
            num_channels)`.
        positive: A 4D tensor containing a batch of images of a different
            identity than the anchor image with shape `(batch_size, height,
            width, num_channels)`.

    It outputs embeddings for each of the images and returns them as a single
    3D tensor of shape `(batch_size, 3, embedding_size)`, where the second
    axis represents the anchor, positive and negative images, respectively.
    The reason for returning the results as a single tensor instead of 3
    separate outputs is because all 3 are required for computing a single loss.

    :return: tf.keras.Model
    """
    base_model = loadModel()
    batch_shape, *input_shape = base_model.input_shape

    anchor_input = Input(input_shape, batch_shape)
    positive_input = Input(input_shape, batch_shape)
    negative_input = Input(input_shape, batch_shape)

    anchor_output = base_model(anchor_input)
    positive_output = base_model(positive_input)
    negative_output = base_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = tf.stack([
        anchor_output,
        positive_output,
        negative_output
    ], axis=1)

    return Model(inputs=inputs, outputs=outputs)


def _maybe_download_weights(path):
    """
    Checks if the VGGFace weights exist, and if not, downloads them.

    :param path: str, the path to where the weights should be stored
    """
    if not os.path.exists(path):
        print("vgg_face_weights.h5 will be downloaded...")
        url = 'https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo'
        gdown.download(url, path, quiet=False)
