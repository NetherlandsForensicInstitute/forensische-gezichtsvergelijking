import os

import dlib
import numpy as np

from deepface.confs.conf import DeepFaceConfs
from deepface.utils.bbox import BoundingBox

from .detector_base import FaceDetector


class FaceDetectorDlib(FaceDetector):
    """
    reference : https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    """
    NAME = 'detector_dlib'

    def __init__(self):
        super(FaceDetectorDlib, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DeepFaceConfs.get()['detector']['dlib']['landmark_detector']
        )
        self.predictor = dlib.shape_predictor(predictor_path)
        self.upsample_scale = DeepFaceConfs.get()['detector']['dlib']['scale']

    def name(self):
        return FaceDetectorDlib.NAME

    def detect(self, npimg):
        dets, scores, idx = self.detector.run(npimg, self.upsample_scale, -1)
        faces = []
        for det, score in zip(dets, scores):
            if score < DeepFaceConfs.get()['detector']['dlib']['score_th']:
                continue

            x = max(det.left(), 0)
            y = max(det.top(), 0)
            w = min(det.right() - det.left(), npimg.shape[1] - x)
            h = min(det.bottom() - det.top(), npimg.shape[0] - y)

            if w <= 1 or h <= 1:
                continue

            bbox = BoundingBox(x, y, w, h, score)

            # find landmark
            bbox.face_landmark = self.detect_landmark(npimg, det)

            faces.append(bbox)

        faces = sorted(faces, key=lambda x: x.score, reverse=True)
        return faces

    def detect_landmark(self, npimg, det):
        shape = self.predictor(npimg, det)
        coords = np.zeros((68, 2), dtype=np.int)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
