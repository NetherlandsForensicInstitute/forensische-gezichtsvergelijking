import abc


class FaceRecognizer(object):
    def __str__(self):
        return self.name()

    @abc.abstractmethod
    def name(self):
        return 'recognizer'

    @abc.abstractmethod
    def extract_features(self, npimg, rois, faces):
        pass

    @abc.abstractmethod
    def detect(self, rois):
        pass

    @abc.abstractmethod
    def get_threshold(self):
        pass
