#%% This Function allows to read files created by OpenFace compare.py




#%% BROWSE FILE
#clear;clc;close all;
#[file,path] = uigetfile('*.txt','Please select a results file');

#fileid = fopen([path '/' file],'r');


from tkinter import filedialog as FileDialog
import numpy as np
import os


file = FileDialog.askopenfilename(
    initialdir=".", 
    filetypes=(
        ("Ficheros de texto", "*.txt"),
        ("Otros ficheros","*.ttt")
    ), 
    title = "Open results file"
)

keyword = '/caras/'

file2 = os.path.basename(file).split('.')[0]

#i = 0
ll = []
fileid = open(file,'r')
for linea in fileid:
      if keyword in linea:
          #encontrado
          
         a = linea.split(':')
         f1 = a[0].split('/')
         f2 = a[1].split('/')
         dist = float(a[2])
         if dist > 0:
             ll.append ([f1[-2] == f2[-2],dist])
            # i +=1
            # if i>40:
            #     break
                 

fileid.close
np.save(file2,ll)

#filename = strsplit(file,'.');
#saveresult = [filename{1},'.mat'];