import pandas as pd

from params import *


class ExperimentSettings:
    """
    Object setting all scorers, calibrators and parameters for a set of experiment runs.
    """

    def __init__(self, args):
        self.calibrators = self.__get_scorer_calibrator_list(args.calibrator, CALIBRATORS)
        self.scorers = self.__get_scorer_calibrator_list(args.scorer, SCORERS)
        self.__set_params(args.params)
        self.__set_data_settings(args.data)

        self.params_dict = dict({'scorers': self.scorers, 'calibrators': self.calibrators})

        self.data_frame = self.set_all_experiments(TIMES)
        self.input_parameters = list(self.data_frame.columns)

    def __get_scorer_calibrator_list(self, run_part, dictionary):
        """
        Function to return the scorers and calibrators to be used based on command line input or the specified
        current_set_up's
        :param run_part: if not None returns listed possibilities in 'run_part' from dictionary
        :param dictionary: dictionary to get data from
        :return: list of objects/values from dictionary
        """
        current_list = dictionary['current_set_up']
        if run_part:
            current_list = run_part
        return [dictionary['all'][i] for i in current_list]

    def __set_params(self, params_cl):
        """
        Filters all parameters in PARAMS for the parameter sets listed in current_set_up or in the list that's passed
        in the command line (params_cl).

        :param params_cl: Contains the parameter set list passed to the command line if any, otherwise None
        """
        param_set_list = PARAMS['current_set_up']
        if params_cl:
            param_set_list = params_cl
        self.parameters = {key: PARAMS['all'][key] for key in param_set_list}

    def __set_data_settings(self, data_cl):
        """
        Filters all data sets in DATA for the data sets listed in current_set_up or in the list that's passed
        in the command line (data_cl)

        :param data_cl: Contains the data set list passed to the command line if any, otherwise None
        """
        data_set_list = DATA['current_set_up']
        if data_cl:
            data_set_list = data_cl
        self.data_settings = {key: DATA['all'][key] for key in data_set_list}

    def set_all_experiments(self, repeat_n_times):
        """
        Returns a Pandas DataFrame of all possible combinations of the parameters in dictionary (incl. scorers
        and calibrators) to input in an experiment.
        Column names are taken from the keys specified in params_dict.
        The total set is repeated repeat_n_times.

        :return: DataFrame
        """
        df = None
        for dat_key in self.data_settings.keys():
            data_dict = self.data_settings[dat_key]
            dictionary_with_data = dict(self.params_dict, **data_dict)
            for par_key in self.parameters.keys():
                par_dict = self.parameters[par_key]
                dictionary_with_params = dict(dictionary_with_data, **par_dict)
                tmp_df = self.all_combinations(dictionary_with_params)
                if df is not None:
                    df = df.append(tmp_df, sort=True)
                else:
                    df = tmp_df
        df['experiment_id'] = range(len(df))
        df = pd.concat([df] * repeat_n_times, ignore_index=True)
        if df is not None:
            df = df.reset_index()
        return df

    def all_combinations(self, dictionary):
        """
        Returns a Pandas DataFrame of all possible combinations of the parameters in dictionary.
        Column names are taken from the keys specified in params_dict.

        Note: Exceptions are built in for repeating/tiling tuple type variables.

        :param: dictionary: Dictionary containing all parameters to be used
        :return: DataFrame
        """
        lengths = [len(e) if type(e) == list else 1 for e in dictionary.values()]
        all_combinations = np.prod(lengths)

        all_combinations_dict = {}
        current_combination_length = 1
        par_names = list(dictionary.keys())
        for i in range(0, len(dictionary)):
            current_combination_length = current_combination_length * lengths[i]
            cur_rep = int(current_combination_length / lengths[i])

            tmp_value = [
                [dictionary[par_names[i]]] if type(dictionary[par_names[i]]) is not list else dictionary[par_names[i]]]
            if not any(isinstance(v, tuple) for v in tmp_value):
                cur_rep_arr = np.repeat(tmp_value, cur_rep)
            else:
                cur_rep_arr = [tuple(v) for v in np.repeat(tmp_value, cur_rep, 0)]

            total_rep = int(all_combinations / len(cur_rep_arr))
            all_combinations_dict[par_names[i]] = np.tile(cur_rep_arr, total_rep) if not any(
                isinstance(v, tuple) for v in cur_rep_arr) else list(cur_rep_arr) * total_rep
        return pd.DataFrame(all_combinations_dict)
