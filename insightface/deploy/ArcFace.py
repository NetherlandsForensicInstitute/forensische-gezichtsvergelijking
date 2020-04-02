import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
import face_model
import argparse
import cv2
import sys
import numpy as np

import urllib
import os
from zipfile import ZipFile 



this_dir = os.getcwd()
models_dir =  os.path.join(this_dir,'insightface','models')
model_name = 'model-r100-ii'
model_dir =  os.path.join(models_dir,model_name)
url = 'https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=1'

# importing required model

if os.path.isdir(model_dir) != True:
    print('Trained model will be downloaded...' + model_dir)
    #output = models_dir + '/downloaded_model.zip'
    zip_file = model_dir +'.zip'
    
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
 
    with open(zip_file, "wb") as f :
        f.write(data)
        f.close()
        
    # opening the zip file in READ mode 
    with ZipFile(zip_file, 'r') as zip: 
     
        # printing all the contents of the zip file 
        zip.printdir() 
      
        # extracting all the files 
        print('Extracting all the files now...') 
        zip.extractall(path= models_dir)
        print('Done!') 
        os.remove(zip_file)


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
#parser.add_argument('--model', default='/home/andrea/insightface/models/model-r100-arcface-ms1m-refine-v2/model,0', help='path to load model.')

parser.add_argument('--model', default=os.path.join(model_dir,'model,0'), help='path to load model.')

parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()



def loadModel():
    model = face_model.FaceModel(args)
    return model









    
          


