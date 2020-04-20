from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
#from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm

from zipfile import ZipFile 
import gdown

cfg_path = 'configs/arc_res50.yaml'

def loadModel():  

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    this_dir = os.path.dirname(__file__)

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(os.path.join(this_dir,cfg_path))

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)
    model_dir =os.path.join(this_dir,'checkpoints',cfg['sub_name'])
    if os.path.isdir(model_dir) != True:
        download_model(model_dir)
    
    
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        return model
        


def download_model(model_dir):
    
    url ='https://drive.google.com/uc?id=1HasWQb86s4xSYy36YbmhRELg9LBmvhvt'
    
    # importing required model

    print('Pre-trained model will be downloaded...' + model_dir)
    #output = models_dir + '/downloaded_model.zip'
    zip_file = model_dir +'.zip' 

    
    gdown.download(url, zip_file , quiet=False)
        
    # opening the zip file in READ mode 
    with ZipFile(zip_file, 'r') as zip: 
     
        # printing all the contents of the zip file 
        zip.printdir() 
      
        # extracting all the files 
        print('Extracting all the files now...') 
        zip.extractall(path= os.path.dirname(model_dir))
        print('Done!') 
        os.remove(zip_file)














        

