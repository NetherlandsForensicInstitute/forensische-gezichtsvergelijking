import os
from zipfile import ZipFile

import gdown
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model
from insightface.modules.models import ArcFaceModel
from insightface.modules.utils import load_yaml

# flags.DEFINE_string('cfg_path', '', 'config file path')
# flags.DEFINE_string('gpu', '0', 'which gpu to use')
# flags.DEFINE_string('img_path', '', 'path to input image')

cfg_path = 'configs/arc_res50.yaml'


def loadModel():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    this_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(this_dir)

    # logger = tf.get_logger()
    # logger.disabled = True
    # logger.setLevel(logging.FATAL)
    # set_memory_growth()

    cfg = load_yaml(os.path.join(parent_dir, cfg_path))

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)
    model_dir = os.path.join(parent_dir, 'weights', cfg['sub_name'])
    if not os.path.isdir(model_dir):
        download_model(model_dir)

    ckpt_path = tf.train.latest_checkpoint(model_dir)

    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)

        inputs = model.input
        embeddings = model.output
        outputs = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embeddings)
        return Model(inputs, outputs)


def download_model(model_dir):
    url = 'https://drive.google.com/uc?id=1HasWQb86s4xSYy36YbmhRELg9LBmvhvt'
    print('Trained model will be downloaded...' + model_dir)
    zip_file = model_dir + '.zip'

    # u = urllib.request.urlopen(url)
    # data = u.read()
    # u.close()

    # with open(zip_file, "wb") as f :
    #     f.write(data)
    #     f.close()

    gdown.download(url, zip_file, quiet=False)

    # opening the zip file in READ mode 
    with ZipFile(zip_file, 'r') as zip:
        # printing all the contents of the zip file 
        zip.printdir()

        # extracting all the files 
        print('Extracting all the files now...')
        zip.extractall(path=os.path.dirname(model_dir))
        print('Done!')
        os.remove(zip_file)
