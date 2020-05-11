from keras_vggface.vggface import VGGFace
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda
import tensorflow as tf


def load_model():
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    embeddings = vgg_model.get_layer('fc7').output
    output = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embeddings)
    custom_vgg_model = Model(vgg_model.input, output)
    return custom_vgg_model
