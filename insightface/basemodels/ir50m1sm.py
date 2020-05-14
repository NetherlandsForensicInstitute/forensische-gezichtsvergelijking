from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import gdown


def prelu_channelwise(x, name=None):
    return PReLU(shared_axes=[1,2], name=name)(x)

def bottleneck_IR(x, in_channel, depth, stride=1, prefix=None):
    if in_channel == depth:
        shortcut = MaxPooling2D((1, 1), stride)(x)
    else:
        shortcut = Conv2D(depth, 1, strides=stride, use_bias=False, name=prefix+".shortcut_layer.0")(x)
        shortcut = BatchNormalization(epsilon=1e-5, name=prefix+".shortcut_layer.1")(shortcut)
        
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=prefix+".res_layer.0")(x)
    x = Conv2D(depth, 3, padding="same", use_bias=False, name=prefix+".res_layer.1")(x)
    x = prelu_channelwise(x, name=prefix+".res_layer.2")
    x = ZeroPadding2D(1)(x)
    x = Conv2D(depth, 3, strides=stride, use_bias=False, name=prefix+".res_layer.3")(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name=prefix+".res_layer.4")(x)
    
    out = Add()([x, shortcut])
    return out

def loadModel():
    inp = Input((112, 112, 3))
    
    # input_layer
    x = Conv2D(64, 3, padding="same", use_bias=False, name="input_layer.0")(inp)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name="input_layer.1")(x)
    x = prelu_channelwise(x, name="input_layer.2")
    
    # body
    # IR_50: [3, 4, 14, 3]; IR_101: [3, 13, 30, 3]; IR_152: [3, 8, 36, 3]
    # blocks 0
    x = bottleneck_IR(x, 64, 64, 2, prefix=f"body.0")
    for i in range(1, 3):
        x = bottleneck_IR(x, 64, 64, prefix=f"body.{str(i)}")
        
    #blocks 1
    x = bottleneck_IR(x, 64, 128, 2, prefix=f"body.3")
    for i in range(4, 7):
        x = bottleneck_IR(x, 128, 128, prefix=f"body.{str(i)}")
        
    # blocks 2
    x = bottleneck_IR(x, 128, 256, 2, prefix=f"body.7")
    for i in range(8, 21):
        x = bottleneck_IR(x, 256, 256, prefix=f"body.{str(i)}")
        
    # blocks 2
    x = bottleneck_IR(x, 256, 512, 2, prefix=f"body.21")
    for i in range(22, 24):
        x = bottleneck_IR(x, 512, 512, prefix=f"body.{str(i)}")    
    
    # output_layer
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name="output_layer.0")(x)
    x = Dropout(0.5)(x, training=False)
    x = Permute((3,1,2))(x)
    x = Flatten()(x)
    x = Dense(512, name="output_layer.3")(x)
    out = BatchNormalization(momentum=0.9, epsilon=1e-5, name="output_layer.4")(x)
    outn = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(out)
    model = Model(inp, outn, name='ir50m1sm')
    this_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(this_dir)
    weights_file = os.path.join(parent_dir, 'weights', 'backbone_ir50_ms1m_keras.h5')
    weights_dir = os.path.dirname(weights_file)
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
    if not os.path.isfile(weights_file):
        print("backbone_ir50_ms1m.h5 will be downloaded...")
        url = 'https://drive.google.com/uc?id=18MyyXQIwhR5I6gzipYMiJ9ywgvFWQMvI'
        gdown.download(url, weights_file, quiet=False)
    model.load_weights(weights_file)
    return model
