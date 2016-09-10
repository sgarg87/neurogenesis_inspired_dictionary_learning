from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
#
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.structure.modules import SoftmaxLayer
#
import scipy.io
import pickle
#
import numpy as np
import numpy.random as npr
#
import sklearn.metrics as skm
#
import pybrain.structure as pys
import math


class NeuralNetworkClassifier:
    def __init__(self, max_iter, num_hidden_layers):
        self.__deep_network_obj__ = None
        self.__train_dataset__ = None
        self.__test_data__ = None
        self.__test_labels__ = None
        self.__trained_model__ = None
        self.__num_features__ = None
        self.__max_iter__ = max_iter
        self.__num_hidden_layers__ = num_hidden_layers

    def initialize_network(self):
        num_features = self.__num_features__
        #
        assert num_features > 5, 'the current network is configured assuming that the number of features at least 5'
        #
        num_hidden_layers = self.__num_hidden_layers__
        # assert (num_hidden_layers % 2) == 1
        assert num_hidden_layers > 0
        #
        num_hidden_nodes_ratio_next_layer = (2**(-8/float(num_hidden_layers+1)))
        print 'num_hidden_nodes_ratio_next_layer', num_hidden_nodes_ratio_next_layer
        #
        num_nodes_in_hidden_layers = []
        curr_layer_num_nodes = num_features
        num_nodes_in_hidden_layers.append(curr_layer_num_nodes)
        for curr_idx in range(num_hidden_layers):
            curr_layer_num_nodes = int(math.ceil(curr_layer_num_nodes*num_hidden_nodes_ratio_next_layer))
            num_nodes_in_hidden_layers.append(curr_layer_num_nodes)
        num_nodes_in_hidden_layers.append(1)
        print 'num_nodes_in_hidden_layers', num_nodes_in_hidden_layers
        num_nodes_in_hidden_layers = num_nodes_in_hidden_layers
        #
        deep_network = \
            buildNetwork(
                *num_nodes_in_hidden_layers
            )
        #
        assert deep_network is not None
        self.__deep_network_obj__ = deep_network

    def __get_supervised_dataset__(self, data):
        number_of_columns = data.shape[1]
        dataset = SupervisedDataSet(number_of_columns-1, 1)
        input_data = data[:, :-1]
        print input_data.shape
        dataset.setField('input', input_data)
        #
        out_data = data[:, -1]
        out_data = out_data.reshape(out_data.size, 1)
        print out_data.shape
        dataset.setField('target', out_data)
        #
        return dataset

    def __get_classification_dataset__(self, data):
        DS = ClassificationDataSet(self.__num_features__, class_labels=['neg', 'pos'])
        for curr_data_idx in range(data.shape[0]):
            curr_data = data[curr_data_idx, :-1]
            print 'curr_data', curr_data
            curr_label = data[curr_data_idx, -1]
            print 'curr_label', curr_label
            DS.appendLinked(curr_data, [curr_label])
        #
        return DS
        # DS.calculateStatistics()
        # print DS.getField('target')
        # print DS.getField('target').shape
        # print DS.getField('input')
        # print DS.getField('input').shape
        # print DS.nClasses
        # print DS.getClass(0)
        # print DS.getClass(1)

    def __get_train_dataset_formatted__(self, data):
        self.__num_features__ = data.shape[1]-1
        #
        return self.__get_supervised_dataset__(data)
        # return self.__get_classification_dataset__(data)

    def set_train_test_datasets(self, data):
        num_data = data.shape[0]
        rand_idx = npr.permutation(num_data)
        data = data[rand_idx, :]
        #
        num_train_data = int(num_data*0.8)
        #
        train_data = data[1:num_train_data, :]
        print 'train_data', train_data.shape
        test_data = data[num_train_data:, :]
        print 'test_data', test_data.shape
        #
        self.__train_dataset__ = self.__get_train_dataset_formatted__(train_data)
        train_data = None
        #
        self.__test_data__ = test_data[:, :-1]
        self.__test_labels__ = test_data[:, -1]

    def train_network(self):
        self.__trained_model__ =\
            BackpropTrainer(
                self.__deep_network_obj__,
                dataset=self.__train_dataset__,
                momentum=0.1,
                verbose=True,
                weightdecay=0.01
            )
        #
        # self.__trained_model__.train()
        self.__trained_model__.trainUntilConvergence(verbose=True, maxEpochs=self.__max_iter__)

    def test_model(self):
        test_data = self.__test_data__
        #
        num_test_data = test_data.shape[0]
        #
        labels = []
        #
        for curr_test_idx in range(num_test_data):
            curr_test_data = test_data[curr_test_idx, :]
            #
            # print 'curr_test_data', curr_test_data
            #
            test_inference = self.__deep_network_obj__.activate(curr_test_data)[0]
            # print 'test_inference', test_inference.tolist()
            #
            if test_inference < 0:
                curr_label = -1
            else:
                curr_label = 1
            #
            labels.append(curr_label)
        #
        confusion_matrix_fr_inference = skm.confusion_matrix(self.__test_labels__, labels)
        #
        print 'confusion_matrix_fr_inference', confusion_matrix_fr_inference
        #
        false_pos_error_rate = confusion_matrix_fr_inference[0, 1]/float(confusion_matrix_fr_inference[:, 1].sum())
        print 'false_pos_error_rate', false_pos_error_rate
        false_neg_error_rate = confusion_matrix_fr_inference[1, 0] / float(confusion_matrix_fr_inference[:, 0].sum())
        print 'false_neg_error_rate', false_neg_error_rate
        totalErr = (confusion_matrix_fr_inference[0, 1] + confusion_matrix_fr_inference[1, 0]) / float(confusion_matrix_fr_inference.sum())
        print 'totalErr', totalErr
        #
        # precision = confusion_matrix_fr_inference[1, 1]/float(confusion_matrix_fr_inference[:, 1].sum())
        # print 'precision', precision
        # recall = confusion_matrix_fr_inference[1, 1]/float(confusion_matrix_fr_inference[1, :].sum())
        # print 'recall', recall
        # f1_score = (2*precision*recall)/(precision+recall)
        # print 'f1_score', f1_score

    def dump(self):
        file_name = './neural_network.pickle'
        #
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)


def load_data(data_file):
    dataMat = scipy.io.loadmat(data_file, mat_dtype=True)
    data = dataMat['data']
    return data


if __name__ == '__main__':
    #
    # data = load_data('./large_image_sparse_codings/mairal_sparse_codings_test.mat')
    data = load_data('./large_image_sparse_codings/neurogen_group_mairal_sparse_codings_test.mat')
    # data = load_data('./large_image_data/large_image_data.mat')
    #
    #
    max_iter = 10
    num_hidden_layers = 3
    #
    nn_obj = \
        NeuralNetworkClassifier(
            max_iter,
            num_hidden_layers
        )
    #
    for curr_idx in range(3):
        print '********************************************'
        nn_obj.set_train_test_datasets(data)
        nn_obj.initialize_network()
        nn_obj.train_network()
        nn_obj.test_model()
    #
    nn_obj.dump()


