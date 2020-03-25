import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import time
import os
import pandas as pd
#from tqdm import tqdm
import json

import face_model
import argparse
import cv2
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


model = face_model.FaceModel(args) #ArcFace model
model_name = 'ArcFace'

def verify(img1_path, img2_path=''
           , distance_metric = 'cosine', plot = False):
	
	tic = time.time()
	
	if type(img1_path) == list:
		bulkProcess = True
		img_list = img1_path.copy()
	else:
		bulkProcess = False
		img_list = [[img1_path, img2_path]]
			
	#------------------------------
	resp_objects = []
	for instance in img_list:
		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]
			img2_path = instance[1]
			
			#----------------------
			
			if os.path.isfile(img1_path) != True:
				raise ValueError("Confirm that ",img1_path," exists")
			
			if os.path.isfile(img2_path) != True:
				raise ValueError("Confirm that ",img2_path," exists")
			
			#----------------------
            
            #Read images 
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
			#crop and align faces
			
			#img1 = functions.detectFace(img1_path, input_shape)
			#img2 = functions.detectFace(img2_path, input_shape)
			
			#----------------------
            
            #Adapt images to model
            
            img1_adapted = model.get_input(img1)
            img2_adapted = model.get_input(img2)
            
			#find embeddings, embeddings are normalized
            
            img1_representation = model.get_feature(img1_adapted)
            img2_representation = model.get_feature(img2_adapted)
	
			#----------------------
			#find distances between embeddings
			
			if distance_metric == 'cosine':
                distance = 1 - np.dot(img1_representation, img2_representation.T)
                threshold = 1 - np.cos(np.arcsin(1.24/2)*2)
			elif distance_metric == 'euclidean':
				distance = np.max(np.abs(img1_representation-img2_representation))
                threshold = np.sin(2*(np.arccos(1.24/2)))
			elif distance_metric == 'euclidean_l2':
                distance = np.sum(np.square(img1_representation-img2_representation))
                threshold = 1.24
			else:
				raise ValueError("Invalid distance_metric passed - ", distance_metric)
			
			#----------------------
			#decision
			
			if distance <= threshold:
				identified =  "true"
			else:
				identified =  "false"
			#----------------------
			if plot:
				label = "Verified: "+identified
				label += "\nThreshold: "+str(round(distance, 2))
				label += ", Max Threshold to Verify: "+str(threshold)
				label += "\nModel: "+model_name
				label += ", Similarity metric: "+distance_metric
				
				fig = plt.figure()
				fig.add_subplot(1,2, 1)
				plt.imshow(img1[0][:, :, ::-1])
				plt.xticks([]); plt.yticks([])
				fig.add_subplot(1,2, 2)
				plt.imshow(img2[0][:, :, ::-1])
				plt.xticks([]); plt.yticks([])
				fig.suptitle(label, fontsize=17)
				plt.show(block=True)
				
			#----------------------
			#response object
			
			resp_obj = "{"
			resp_obj += "\"verified\": "+identified
			resp_obj += ", \"distance\": "+str(distance)
			resp_obj += ", \"max_threshold_to_verify\": "+str(threshold)
			resp_obj += ", \"model\": \""+model_name+"\""
			resp_obj += ", \"similarity_metric\": \""+distance_metric+"\""
			resp_obj += "}"
			
			resp_obj = json.loads(resp_obj) #string to json
			
			if bulkProcess == True:
				resp_objects.append(resp_obj)
			else:
				return resp_obj
			#----------------------
			
		else:
			raise ValueError("Invalid arguments passed to verify function: ", instance)
		
	#-------------------------
	
	toc = time.time()
	
	#print("identification lasts ",toc-tic," seconds")
	
	if bulkProcess == True:
		return resp_objects

def analyze(img_path, actions= []):
	
	if type(img_path) == list:
		img_paths = img_path.copy()
		bulkProcess = True
	else:
		img_paths = [img_path]
		bulkProcess = False
	
	#---------------------------------
	

