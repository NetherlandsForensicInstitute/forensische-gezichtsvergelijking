import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/andrea/insightface/models/model-r100-arcface-ms1m-refine-v2/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()



def loadModel():
    model = face_model.FaceModel(args)
    return model

