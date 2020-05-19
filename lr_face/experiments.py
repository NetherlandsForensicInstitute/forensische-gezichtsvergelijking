from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Iterator, Tuple, Optional, Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from lr_face.data import Dataset, split_by_identity, make_pairs, FacePair, \
    FaceImage, make_pairs_from_two_lists
from lr_face.models import ScorerModel
from lr_face.versioning import Tag
from params import *


@dataclass
class Experiment:
    data_config: Dict[str, Any]
    scorer: ScorerModel
    calibrator: BaseEstimator
    params: Dict[str, Any]

    def __str__(self):
        """
        Converts the configuration of this experiment to a string that can be
        used to generate file names for example.
        """
        data_values = []
        for k, v in self.data_config.items():
            if k == 'datasets' and isinstance(v, tuple):
                data_values.append('|'.join(map(str, v)))
            else:
                data_values.append(str(v))

        data_str = '_'.join(data_values)
        params_str = '_'.join(map(str, self.params.values()))
        return '_'.join(map(str, [
            self.scorer,
            self.calibrator,
            data_str,
            params_str
        ])).replace(':', '-')  # Windows forbids ':'

    def get_calibration_and_test_pairs(self) -> Tuple[
        Dict[Tuple, List[FacePair]],
        Dict[Tuple, List[FacePair]]
    ]:
        assert isinstance(self.data_config['calibration'], tuple)
        assert isinstance(self.data_config['test'], tuple)

        # get all images
        calibration_images = []
        for dataset in self.data_config['calibration']:
            calibration_images += dataset.images

        # filter the images per category
        calibration_images_per_category = defaultdict(list)
        for image in calibration_images:
            calibration_images_per_category[
                tuple(getattr(image, prop)
                 for prop in self.data_config['calibration_filters'])]\
                .append(image)

        calibration_pairs_per_category = {}
        for category_a, images_a in calibration_images_per_category.items():
            for category_b, images_b in \
                    calibration_images_per_category.items():
                calibration_pairs_per_category[(category_a, category_b)] = \
                    make_pairs_from_two_lists(images_a, images_b)

        test_pairs = []
        for dataset in self.data_config['test']:
            test_pairs += dataset.pairs
        test_pair_categories = [(
            (getattr(pair.first, prop)
                 for prop in self.data_config['calibration_filters']),
            (getattr(pair.second, prop)
                 for prop in self.data_config['calibration_filters']))
            for pair in test_pairs]

        test_pairs_per_category = defaultdict(list)
        for category, pair in zip(test_pair_categories, test_pairs):
            test_pairs_per_category[category].append(pair)

        return calibration_pairs_per_category, test_pairs_per_category



class ExperimentalSetup:
    def __init__(self,
                 scorer_names: List[str],
                 calibrator_names: List[str],
                 data_config_names: List[str],
                 param_names: List[str],
                 num_repeats: int):
        self.scorers = self._get_scorers(scorer_names)
        self.calibrators = self._get_calibrators(calibrator_names)
        self.data_config = self._get_data_config(data_config_names)
        self.params = self._get_params(param_names)
        self.num_repeats = num_repeats
        self.name = datetime.now().strftime("%Y-%m-%d %H %M %S")
        self.experiments = self.prepare_experiments()

    def prepare_experiments(self) -> List[Experiment]:
        """
        Returns a list of all experiments that fall under this setup.

        :return: List[Experiment]
        """
        experiments = []
        for scorer in self.scorers:
            for calibrator in self.calibrators:
                for data_config in self.data_config:
                    for params in self.params:
                        experiments.append(Experiment(
                            data_config,
                            scorer,
                            calibrator,
                            params
                        ))
        return experiments * self.num_repeats

    def __iter__(self) -> Iterator[Experiment]:
        return iter(self.experiments)

    def __len__(self) -> int:
        return len(self.experiments)

    @staticmethod
    def _get_calibrators(calibrator_names: Optional[List[str]] = None) \
            -> List[BaseEstimator]:
        """
        Parses a list of CALIBRATORS configuration names and returns the
        corresponding calibrators. If no names are given, the ones specified
        under `CALIBRATORS['current_set_up']` are used.

        :param calibrator_names: List[str]
        :return: List[ScorerModel]
        """
        if not calibrator_names:
            calibrator_names = CALIBRATORS['current_set_up']
        return [CALIBRATORS['all'][c] for c in calibrator_names]

    @staticmethod
    def _get_scorers(scorer_names: Optional[List[str]] = None) \
            -> List[ScorerModel]:
        """
        Parses a list of SCORERS configuration names and returns the
        corresponding `ScorerModel` instances. If no names are given, the ones
        specified under `SCORERS['current_set_up']` are used.

        :param scorer_names: List[str]
        :return: List[ScorerModel]
        """
        if not scorer_names:
            scorer_names = SCORERS['current_set_up']

        def init_scorer(architecture: Architecture,
                        tag: Optional[Union[str, Tag]]) -> ScorerModel:
            if isinstance(tag, str):
                tag = Tag(tag)
            # If no version is specified, use latest version.
            if tag and not tag.version:
                tag.version = architecture.get_latest_version(tag)
            return architecture.get_scorer_model(tag)

        return [init_scorer(*SCORERS['all'][s]) for s in scorer_names]

    @staticmethod
    def _get_params(param_names: Optional[List[str]] = None) \
            -> List[Dict[str, Any]]:
        """
        Parses a list of PARAMS configuration names and returns the
        corresponding PARAMS configurations. If no names are given, the ones
        specified under `PARAMS['current_set_up']` are used.

        :param param_names: List[str]
        :return: List[Dict[str, Any]]
        """
        if not param_names:
            param_names = PARAMS['current_set_up']
        return [PARAMS['all'][key] for key in param_names]

    @staticmethod
    def _get_data_config(data_config_names: Optional[List[str]] = None) \
            -> List[Dict[str, Any]]:
        """
        Parses a list of DATA configuration names and returns the corresponding
        DATA configurations. If no names are given, the ones specified under
        `DATA['current_set_up']` are used.

        :param data_config_names: List[str]
        :return: List[Dict[str, Any]]
        """
        if not data_config_names:
            data_config_names = DATA['current_set_up']
        return [DATA['all'][key] for key in data_config_names]

    @property
    def params_keys(self) -> List[str]:
        """
        Returns all keys that need to be specified for a valid PARAMS
        configuration.

        :return: List[str]
        """
        return list(set(k for v in PARAMS['all'].values() for k in v.keys()))

    @property
    def data_keys(self) -> List[str]:
        """
        Returns all keys that need to be specified for a valid DATA
        configuration.

        :return: List[str]
        """
        return list(set(k for v in DATA['all'].values() for k in v.keys()))
