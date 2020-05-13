import json
import argparse
import os
import re
from glob import glob


def parser():
    arg_parser = argparse.ArgumentParser(
        description='Run preprocessing on a folder with images')

    arg_parser.add_argument(
        '--input_folder',
        '-i',
        type=str,
        action='store',
        required=True,
        help='input folder, contains contains one or more label-studio annotation folders, or a completion folder'
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


def yaw_rename(yaw):
    d = {"recht": "straight",
         "licht gedraaid": "slightly_turned",
         "ver gedraaid": "sideways" }
    return d[yaw]


def pitch_rename(pitch):
    d = {"ver naar boven": "upwards",
         "licht naar boven": "slightly_upwards",
         "recht": "straight",
         "licht naar onder": "slightly_downwards",
         "ver naar onder": "downwards"}
    return d[pitch]


def parse_annotation(annotation_path):
    with open(annotation_path) as annotation_json:
        annotation = json.load(annotation_json)

        # put default values in the dict
        annotation_dict = {"yaw": None,
                           "pitch": None,
                           "headgear": False,
                           "glasses": False,
                           "beard": False,
                           "other_occlusions": False,
                           "low_quality": False,
                           "path": os.path.join("resources", os.path.relpath(annotation["task_path"], "resources"))}

        for sub_task in annotation["completions"][0]["result"]:
            if sub_task["from_name"] == "yaw":
                annotation_dict["yaw"] = yaw_rename(sub_task["value"]["choices"][0])

            if sub_task["from_name"] == "pitch":
                annotation_dict["pitch"] = pitch_rename(sub_task["value"]["choices"][0])

            if sub_task["from_name"] == "overig":
                other_choices = sub_task["value"]["choices"]

                if "hoofddeksel" in other_choices:
                    annotation_dict["headgear"] = True

                if "bril" in other_choices:
                    annotation_dict["glasses"] = True

                if "baard" in other_choices:
                    annotation_dict["beard"] = True

                if "occlusion overig" in other_choices:
                    annotation_dict["other_occlusions"] = True

                if "slechte kwaliteit" in other_choices:
                    annotation_dict["low_quality"] = True

    return annotation_dict


def write_json(annotation_dict):
    path = annotation_dict["path"]
    json_path = path.replace(path, re.sub(r'\.[^.]+$', '.json', path))
    with open(json_path, 'w') as f:
        f.write(json.dumps(annotation_dict, indent=4))


def run(input_folder, recursive):
    paths = glob(input_folder + '/**/completions/*.json',
                 recursive=True) if recursive else glob(input_folder + '/*')

    for path in paths:
        annotation_dict = parse_annotation(path)
        write_json(annotation_dict)


if __name__ == '__main__':
    '''
    Example run script: process_annotations.py -i "annotations" -r
    '''
    pars = parser()
    args = pars.parse_args()
    run(**vars(args))


