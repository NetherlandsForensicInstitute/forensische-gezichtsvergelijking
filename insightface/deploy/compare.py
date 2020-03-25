import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-arcface-ms1m-refine-v2/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)


def Comparison( img1, img2):
    img = cv2.imread(img1)
    img = model.get_input(img)
    f1 = model.get_feature(img)
    
    img = cv2.imread(img2)
    img = model.get_input(img)
    f2 = model.get_feature(img)   
    
    dist12 = np.sum(np.square(f1-f2))
    sim12 = np.dot(f1, f2.T)

    return([dist12,sim12])
    

