import constants_dict_methods as cdm
import data_sets
import init_parameters
import dictionary_learning as dl
import constants_parameter_names as cpn


class ModelEvaluation:
    def __init__(self):
        pass

    def evaluate_model(self):
        pass

    def init_data(self, T, input_dim):
        train_data, test_data, data0, test_data0, data1, test_data1 =\
            data_sets.get_large_images(T, input_dim)
        #
        datasets_map = {}
        datasets_map['data_nst2_train'] = data1
        datasets_map['data_nst2_test'] = test_data1
        datasets_map['data_st_train'] = train_data
        datasets_map['data_st_test'] = test_data
        datasets_map['data_nst_train'] = data0
        datasets_map['data_nst_test'] = test_data0
        #
        self.datasets_map = datasets_map
        datasets_map = None

    def init_dictionary_sizes(self):
        global_size = [25, 150, 500]
        #
        dictionary_sizes = {}
        #
        if self.algorithms[cdm.mairal]:
            dictionary_sizes[cdm.mairal] = global_size
        #
        if self.algorithms[cdm.random]:
            dictionary_sizes[cdm.random] = dictionary_sizes[cdm.mairal]
        #
        if self.algorithms[cdm.group_mairal]:
            dictionary_sizes[cdm.group_mairal] = global_size
        #
        if self.algorithms[cdm.sg]:
            dictionary_sizes[cdm.sg] = global_size
        #
        if self.algorithms[cdm.neurogen_mairal]:
            dictionary_sizes[cdm.neurogen_mairal] = global_size
        #
        if self.algorithms[cdm.neurogen_sg]:
            dictionary_sizes[cdm.neurogen_sg] = global_size
        #
        if self.algorithms[cdm.neurogen_group_mairal]:
            dictionary_sizes[cdm.neurogen_group_mairal] = global_size
        #
        global_size = None
        self.dictionary_sizes = dictionary_sizes
        dictionary_sizes = None

    def init_list_evaluation_algorithms(self):
        #
        algorithms = {}
        #
        algorithms[cdm.mairal] = False
        algorithms[cdm.random] = False
        algorithms[cdm.group_mairal] = False
        algorithms[cdm.neurogen_group_mairal] = False
        algorithms[cdm.neurogen_mairal] = False
        algorithms[cdm.neurogen_sg] = False
        algorithms[cdm.sg] = False
        #
        self.algorithms = algorithms

    # def get_evaluation_parameter_settings(self):
    #     pass

    def init_params(self):
        self.params = init_parameters.init_parameters()

    def init_model(self):
        self.init_params()
        self.init_dictionary_sizes()
        self.init_list_evaluation_algorithms()
        self.init_data()
        #
        dict_models = {}
        #
        for curr_algo in self.algorithms:
            if self.algorithms[curr_algo]:
                print curr_algo
                #
                assert curr_algo in self.dictionary_sizes
                curr_algo_dict_sizes = self.dictionary_sizes[curr_algo]
                #
                curr_algo_dict_models = {}
                #
                for curr_dict_size in curr_algo_dict_sizes:
                    curr_dict_model =\
                        dl.DictionaryLearning(
                            alg=curr_algo,
                            params=params
                        )
                    curr_algo_dict_models[curr_dict_size] = curr_dict_model
                #
                dict_models[curr_algo] = curr_algo_dict_models
        #
        self.dict_models = dict_models

    def adapt_model(self):
        pass

    def plot_evaluation(self):
        pass


