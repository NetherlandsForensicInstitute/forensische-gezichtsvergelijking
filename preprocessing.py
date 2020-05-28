import argparse
import os
from collections import defaultdict
from glob import glob

import cv2

from deepface.commons import functions

import face_recognition


def parser():
    arg_parser = argparse.ArgumentParser(
        description='Run preprocessing on a folder with images')

    arg_parser.add_argument(
        '--input_folder',
        '-i',
        type=str,
        action='store',
        required=True,
        help='input folder, contains the images to be converted'
    )
    arg_parser.add_argument(
        '--output_folder',
        '-o',
        type=str,
        action='store',
        required=True,
        help='location where the converted images should be written to'
    )
    arg_parser.add_argument(
        '--recursive',
        '-r',
        action='store_const',
        const=True,
        default=False,
        help='Apply script recursively'
    )
    return arg_parser


def is_image(file):
    _, ext = os.path.splitext(file)
    return ext.lower() in {'.jpg', '.jpeg', '.png'}


def run(input_folder, output_folder, recursive):
    paths = glob(input_folder + '/**/*.*',
                 recursive=True) if recursive else glob(input_folder + '/*')
    paths = filter(is_image, paths)
    meta = defaultdict(list)
    for path in paths:
        print(path)
        try:
            face, rotation, face_found, original_res = \
                functions.detectFace(path)
            face_found2 = find_face2(path)
            output_path = os.path.join(output_folder, path)
            dir_path, file_name = os.path.split(output_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            cv2.imwrite(output_path, face[0] * 255)
            # save meta info per folder
            meta[dir_path].append(
                [file_name, round(rotation, 2), face_found, face_found2, original_res])
        except ValueError as e:
            print(e)
            pass
    for folder in meta:
        with open(os.path.join(folder, "meta.txt"), "w") as file:
            file.write('file;rotation;face found;face_found2;original resolution\n')
            file.writelines(';'.join(map(str, line)) + '\n' for line in
                            sorted(meta[folder], key=lambda x: x[0]))


def find_face2(path):
    # image = face_recognition.load_image_file(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        return True
    else:
        return False


if __name__ == '__main__':
    '''
    Example runscript: preprocessing.py -i "resources" -o "resizedeepface" -r
    '''
    pars = parser()
    args = pars.parse_args()
    run(**vars(args))
