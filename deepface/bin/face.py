from __future__ import absolute_import

import logging
import os
import pickle
import sys
from glob import glob

import cv2
import numpy as np

import fire
from sklearn.metrics import roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from deepface.confs.conf import DeepFaceConfs
from deepface.detectors.detector_dlib import FaceDetectorDlib
from deepface.detectors.detector_ssd import FaceDetectorSSDMobilenetV2, FaceDetectorSSDInceptionV2
from deepface.recognizers.recognizer_vgg import FaceRecognizerVGG
from deepface.recognizers.recognizer_resnet import FaceRecognizerResnet
from deepface.utils.common import get_roi, feat_distance_l2, feat_distance_cosine
from deepface.utils.visualization import draw_bboxs

logger = logging.getLogger('DeepFace')
logger.setLevel(logging.INFO if int(os.environ.get('DEBUG', 0)) == 0 else logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


class DeepFace:
    def __init__(self):
        self.detector = None
        self.recognizer = None

    def set_detector(self, detector):
        if self.detector is not None and self.detector.name() == detector:
            return
        logger.debug('set_detector old=%s new=%s' % (self.detector, detector))
        if detector == FaceDetectorDlib.NAME:
            self.detector = FaceDetectorDlib()
        elif detector == 'detector_ssd_inception_v2':
            self.detector = FaceDetectorSSDInceptionV2()
        elif detector == 'detector_ssd_mobilenet_v2':
            self.detector = FaceDetectorSSDMobilenetV2()

    def set_recognizer(self, recognizer):
        if self.recognizer is not None and self.recognizer.name() == recognizer:
            return
        logger.debug('set_recognizer old=%s new=%s' % (self.recognizer, recognizer))
        if recognizer == FaceRecognizerVGG.NAME:
            self.recognizer = FaceRecognizerVGG()
        elif recognizer == FaceRecognizerResnet.NAME:
            self.recognizer = FaceRecognizerResnet()

    def blackpink(self, visualize=True):
        imgs = ['./samples/blackpink/blackpink%d.jpg' % (i + 1) for i in range(7)]
        for img in imgs:
            self.run(image=img, visualize=visualize)

    def recognizer_test_run(self, detector=FaceDetectorDlib.NAME, recognizer=FaceRecognizerResnet.NAME, image='./samples/ajb.jpg', visualize=False):
        self.set_detector(detector)
        self.set_recognizer(recognizer)

        if isinstance(image, str):
            logger.debug('read image, path=%s' % image)
            npimg = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            npimg = image
        else:
            logger.error('Argument image should be str or ndarray. image=%s' % str(image))
            sys.exit(-1)

        if npimg is None:
            logger.error('image can not be read, path=%s' % image)
            sys.exit(-1)

        if recognizer:
            logger.debug('run face recognition+')
            result = self.recognizer.detect([npimg[...,::-1]])
            logger.debug('run face recognition-')
        return

    def run_recognizer(self, npimg, faces, recognizer=FaceRecognizerResnet.NAME):
        self.set_recognizer(recognizer)
        rois = []
        for face in faces:
            # roi = npimg[face.y:face.y+face.h, face.x:face.x+face.w, :]
            roi = get_roi(npimg, face, roi_mode=recognizer)
            if int(os.environ.get('DEBUG_SHOW', 0)) == 1:
                cv2.imshow('roi', roi)
                cv2.waitKey(0)
            rois.append(roi)
            face.face_roi = roi

        if len(rois) > 0:
            logger.debug('run face recognition+')
            result = self.recognizer.detect(rois=rois, faces=faces)
            logger.debug('run face recognition-')
            for face_idx, face in enumerate(faces):
                face.face_feature = result['feature'][face_idx]
                logger.debug('candidates: %s' % str(result['name'][face_idx]))
                if result['name'][face_idx]:
                    name, score = result['name'][face_idx][0]
                    # if score < self.recognizer.get_threshold():
                    #     continue
                    face.face_name = name
                    face.face_score = score
        return faces

    def run(self, detector='detector_ssd_mobilenet_v2', recognizer=FaceRecognizerResnet.NAME, image='./samples/blackpink/blackpink1.jpg',
            visualize=False):
        self.set_detector(detector)
        self.set_recognizer(recognizer)

        if image is None:
            return []
        elif isinstance(image, str):
            logger.debug('read image, path=%s' % image)
            npimg = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            npimg = image
        else:
            logger.error('Argument image should be str or ndarray. image=%s' % str(image))
            sys.exit(-1)

        if npimg is None:
            logger.error('image can not be read, path=%s' % image)
            sys.exit(-1)

        logger.debug('run face detection+ %dx%d' % (npimg.shape[1], npimg.shape[0]))
        faces = self.detector.detect(npimg)

        logger.debug('run face detection- %s' % len(faces))

        if recognizer:
            faces = self.run_recognizer(npimg, faces, recognizer)

        img = draw_bboxs(np.copy(npimg), faces)
        cv2.imwrite('result.jpg', img)
        if visualize and visualize not in ['false', 'False']:
            cv2.imshow('DeepFace', img)
            cv2.waitKey(0)

        return faces

    def save_and_run(self, path, image, visualize=True):
        """
        :param visualize:
        :param path: samples/faces
        :param image_path: samples/blackpink1.jpg
        :return:
        """
        self.save_features_path(path)
        self.run(image=image, visualize=visualize)

    def save_features_path(self, path="./samples/blackpink/faces/"):
        """

        :param path: folder contain images("./samples/faces/")
        :return:
        """
        name_paths = [(os.path.basename(img_path)[:-4], img_path)
                      for img_path in glob(os.path.join(path, "*.jpg"))]

        features = {}
        for name, path in tqdm(name_paths):
            logger.debug("finding faces for %s:" % path)
            faces = self.run(image=path)
            features[name] = faces[0].face_feature

        import pickle
        with open('db.pkl', 'wb') as f:
            pickle.dump(features, f, protocol=2)

    def test_lfw(self, set='test', model='ssdm_resnet', visualize=True):
        if set is 'train':
            pairfile = 'pairsDevTrain.txt'
        else:
            pairfile = 'pairsDevTest.txt'
        lfw_path = DeepFaceConfs.get()['dataset']['lfw']
        path = os.path.join(lfw_path, pairfile)
        with open(path, 'r') as f:
            lines = f.readlines()[1:]

        pairs = []
        for line in lines:
            elms = line.split()
            if len(elms) == 3:
                pairs.append((elms[0], int(elms[1]), elms[0], int(elms[2])))
            elif len(elms) == 4:
                pairs.append((elms[0], int(elms[1]), elms[2], int(elms[3])))
            else:
                logger.warning('line should have 3 or 4 elements, line=%s' % line)

        detec = FaceDetectorDlib.NAME
        if model == 'baseline':
            recog = FaceRecognizerVGG.NAME
            just_name = 'vgg'
        elif model == 'baseline_resnet':
            recog = FaceRecognizerResnet.NAME
            just_name = 'resnet'
        elif model == 'ssdm_resnet':
            recog = FaceRecognizerResnet.NAME
            just_name = 'resnet'
            detec = 'detector_ssd_mobilenet_v2'
        else:
            raise Exception('invalid model name=%s' % model)

        logger.info('pair length=%d' % len(pairs))
        test_result = []  # score, label(1=same)
        for name1, idx1, name2, idx2 in tqdm(pairs):
            img1_path = os.path.join(lfw_path, name1, '%s_%04d.jpg' % (name1, idx1))
            img2_path = os.path.join(lfw_path, name2, '%s_%04d.jpg' % (name2, idx2))
            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
            img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

            if img1 is None:
                logger.warning('image not read, path=%s' % img1_path)
            if img2 is None:
                logger.warning('image not read, path=%s' % img2_path)

            result1 = self.run(image=img1, detector=detec, recognizer=recog, visualize=False)
            result2 = self.run(image=img2, detector=detec, recognizer=recog, visualize=False)

            if len(result1) == 0:
                logger.warning('face not detected, name=%s(%d)! %s(%d)' % (name1, idx1, name2, idx2))
                test_result.append((0.0, name1 == name2))
                continue
            if len(result2) == 0:
                logger.warning('face not detected, name=%s(%d) %s(%d)!' % (name1, idx1, name2, idx2))
                test_result.append((0.0, name1 == name2))
                continue

            feat1 = result1[0].face_feature
            feat2 = result2[0].face_feature
            similarity = feat_distance_cosine(feat1, feat2)
            test_result.append((similarity, name1 == name2))

        # calculate accuracy TODO
        accuracy = sum([label == (score > DeepFaceConfs.get()['recognizer'][just_name]['score_th']) for score, label in test_result]) / float(len(test_result))
        logger.info('accuracy=%.8f' % accuracy)

        # ROC Curve, AUC
        tps = []
        fps = []
        accuracy0 = []
        accuracy1 = []
        acc_th = []

        for th in range(0, 100, 5):
            th = th / 100.0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for score, label in test_result:
                if score >= th and label == 1:
                    tp += 1
                elif score >= th and label == 0:
                    fp += 1
                elif score < th and label == 0:
                    tn += 1
                elif score < th and label == 1:
                    fn += 1
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            tps.append(tpr)
            fps.append(fpr)
            accuracy0.append(tn / (tn + fp + 1e-12))
            accuracy1.append(tp / (tp + fn + 1e-12))
            acc_th.append(th)

        fpr, tpr, thresh = roc_curve([x[1] for x in test_result], [x[0] for x in test_result])
        fnr = 1 - tpr
        eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        logger.info('1-eer=%.4f' % (1.0 - eer))

        with open('./etc/test_lfw.pkl', 'rb') as f:
            results = pickle.load(f)

        if visualize in [True, 'True', 'true', 1, '1']:
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            plt.title('Experiment on LFW')
            plt.plot(fpr, tpr, label='%s(%.4f)' % (model, 1 - eer))  # TODO : label

            for model_name in results:
                if model_name == model:
                    continue
                fpr_prev = results[model_name]['fpr']
                tpr_prev = results[model_name]['tpr']
                eer_prev = results[model_name]['eer']
                plt.plot(fpr_prev, tpr_prev, label='%s(%.4f)' % (model_name, 1 - eer_prev))

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            a.legend()
            a.set_title('Receiver operating characteristic')

            a = fig.add_subplot(1, 2, 2)
            plt.plot(accuracy0, acc_th, label='Accuracy_diff')
            plt.plot(accuracy1, acc_th, label='Accuracy_same')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            a.legend()
            a.set_title('%s : TP, TN' % model)

            fig.savefig('./etc/roc.png', dpi=300)
            plt.show()
            plt.draw()

        with open('./etc/test_lfw.pkl', 'wb') as f:
            results[model] = {
                'fpr': fpr,
                'tpr': tpr,
                'acc_th': acc_th,
                'accuracy0': accuracy0,
                'accuracy1': accuracy1,
                'eer': eer
            }
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

        return 1.0 - eer


if __name__ == '__main__':
    fire.Fire(DeepFace)
