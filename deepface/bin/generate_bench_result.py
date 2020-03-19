import os
import sys
from glob import glob

import cv2
import fire
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from deepface import get_detector, get_recognizer, save_features
from deepface.utils.visualization import draw_bboxs
from deepface.utils.common import tag_faces


def show_with_face(npimg, faces, visualize=False):
    if visualize:
        img = draw_bboxs(np.copy(npimg), faces)
        cv2.imshow('DeepFace', img)
        cv2.waitKey(0)


class DeepFace:
    def __init__(self):
        pass

    def run(self,
            folder_path=None,
            detector_method='dlib',
            visualize=True):

        # detect
        detector = get_detector(name='ssd_mobilenet_v2')

        print(os.path.join(folder_path, "*.jpg"))
        for img_path in glob(os.path.join(folder_path, "*.jpg")):
            npimg = cv2.imread(img_path, cv2.IMREAD_COLOR)
            faces = detector.detect(npimg=npimg)

            print("![face](%s)" % img_path)
            for face in faces:
                print(face, face.score, face.face_landmark)

            show_with_face(npimg, faces, visualize=visualize)

    def generate_fddb_ret(self,
                          detector_method='ssd_mobilenet_v2',
                          output_fddb_ret_txt='fddb_ret.txt',
                          img_list_path="/data/public/rw/datasets/faces/fddb/filePath.txt"):

        detector = get_detector(name=detector_method)

        with open(img_list_path, "r") as fr:
            img_list = [img_path.strip() for img_path in fr]

        base_dir = "/data/public/rw/datasets/faces/fddb/originalPics"

        with open(output_fddb_ret_txt, "w") as fw:
            for img_path in img_list:
                real_img_path = os.path.join(base_dir, img_path+".jpg")
                print(real_img_path+ "\n\n\n")

                npimg = cv2.imread(real_img_path, cv2.IMREAD_COLOR)
                faces = detector.detect(npimg=npimg)

                fw.write(img_path + "\n")
                fw.write("%d\n" % len(faces))
                for face in faces:
                    fw.write("%d %d %d %d %f\n" % (face.x, face.y, face.w, face.h, face.score))


if __name__ == '__main__':
    fire.Fire(DeepFace)
