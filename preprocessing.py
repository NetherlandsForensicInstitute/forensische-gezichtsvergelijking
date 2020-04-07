import argparse
import os
from glob import glob
import cv2

from deepface.deepface.commons import functions


def parser():
    arg_parser = argparse.ArgumentParser(description='Run preprocessing on a folder with images')

    arg_parser.add_argument('--input_folder', '-i', type=str, action='store', required=True,
                            help='input folder, contains the images to be converted')
    arg_parser.add_argument('--output_folder', '-o', type=str, action='store', required=True,
                            help='location where the converted images should be written to')
    arg_parser.add_argument('--dimensions', '-d', type=int, action='store', required=True, nargs=2,
                            help='dimensions the fotos should be resized to')
    arg_parser.add_argument('--recursive', '-r', action='store_const', const=True, default=False,
                            help='Apply script recursively')

    return arg_parser


def is_image(file):
    _, ext = os.path.splitext(file)
    return ext.lower() in {'.jpg', '.jpeg', '.png'}


def run(input_folder, output_folder, dimensions, recursive):
    paths = glob(input_folder + '/**/*.*', recursive=True) if recursive else glob(input_folder + '/*')
    paths = filter(is_image, paths)
    for path in paths:
        try:
            face = functions.detectFace(path, target_size=tuple(dimensions))
            output_path = os.path.join(output_folder, path)
            dir_path = os.path.dirname(output_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            cv2.imwrite(output_path, face[0] * 255)
        except ValueError as e:
            print(e)
            pass


if __name__ == '__main__':
    '''
    Example runscript: preprocessing.py -i "resources" -o "resizedeepface" -d 152 152 -r
    '''
    pars = parser()
    args = pars.parse_args()
    run(**vars(args))
