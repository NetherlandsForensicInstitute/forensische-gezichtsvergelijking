import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

from keras_vggface.vggface import VGGFace


def loadModel():
    vgg_model = VGGFace(
        include_top=True, model='resnet50', input_shape=(224, 224, 3))
    embeddings = vgg_model.layers[-2].output
    output = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embeddings)
    return Model(vgg_model.input, output)
