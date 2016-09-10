import codecs
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.sparse as ss
import re
import save_sparse_scipy_matrices as sssm
import scipy.io as si


hn_wordvectors_data = './hn_wordvectors_data'


def load_object():
    with open(hn_wordvectors_data + '.pickle', 'rb') as f:
        obj = pickle.load(f)
        return obj


class ProcessTextFileDataFrWordVec:
    def __init__(self):
        self.vocab = None
        self.data_arr = None
        self.data_idx_in_vocab = None
        self.data_adjacency_matrix = None

    def read_data(self, file_path):
        with codecs.open(file_path, encoding='utf8') as f:
            data_lines = f.readlines()
        num_data_lines = len(data_lines)
        #
        # print 'num_data_lines', num_data_lines
        #
        data_list = []
        for curr_data in data_lines:
            # print '**********************'
            # print curr_data
            curr_data = re.sub('[^a-zA-Z ]', '', curr_data)
            if not curr_data:
                continue
            else:
                curr_data = curr_data.lower()
            #
            # print curr_data
            #
            curr_data_words = curr_data.split()
            if len(curr_data_words) > 3:
                data_list += curr_data_words
        data_lines = None
        num_data = len(data_list)
        #
        # print 'num_data', num_data
        #
        data_arr = np.array(data_list)
        data_list = None
        return data_arr

    def get_vocab(self, data_arr):
        # print data_arr
        vocab = []
        for curr_word in data_arr:
            if curr_word not in vocab:
                vocab.append(curr_word)
        num_vocab = len(vocab)
        print 'num_vocab', num_vocab
        #
        vocab = np.array(vocab)
        print 'vocab', vocab.tolist()
        #
        return vocab

    def update_data_stream_wd_vocab_idx(self):
        # replace the data stream with vocab index.
        #
        data_idx_in_vocab = -1*np.ones(self.data_arr.size, dtype=int)
        #
        for curr_word_idx_in_vocab, curr_word in np.ndenumerate(self.vocab):
            curr_word_idx_in_data = np.where(self.data_arr == curr_word)[0]
            data_idx_in_vocab[curr_word_idx_in_data] = curr_word_idx_in_vocab
        #
        assert np.all(data_idx_in_vocab >= 0)
        self.data_idx_in_vocab = data_idx_in_vocab

    def save_object(self):
        with open(hn_wordvectors_data+'.pickle', 'wb') as f:
            pickle.dump(self, f)

    def compute_data_matrix(self):
        num_vocab = self.vocab.size
        num_data = self.data_idx_in_vocab.size
        #
        data_adjacency_matrix = ss.dok_matrix((num_vocab, num_vocab), dtype=float)
        #
        neighbor_size = 5
        #
        for curr_idx in range(num_data):
            curr_idx_prv = max(curr_idx-neighbor_size, 0)
            curr_idx_nxt = min(curr_idx+neighbor_size, num_data-1)
            #
            curr_data_idx = self.data_idx_in_vocab[curr_idx]
            # curr_idx = None
            #
            curr_neighborhood_data_idx = self.data_idx_in_vocab[curr_idx_prv:curr_idx_nxt+1]
            curr_idx_prv = None
            curr_idx_nxt = None
            #
            print '.........................................'
            print 'curr_idx', curr_idx
            print 'curr_data_idx', curr_data_idx
            print 'curr_data', self.vocab[curr_data_idx]
            print 'curr_neighborhood_data_idx', curr_neighborhood_data_idx
            print 'curr_neighborhood_data', self.vocab[curr_neighborhood_data_idx]
            #
            for curr_neighbor in curr_neighborhood_data_idx:
                data_adjacency_matrix[curr_data_idx, curr_neighbor] += 1
                print 'data_adjacency_matrix[curr_data_idx, curr_neighbor]', data_adjacency_matrix[curr_data_idx, curr_neighbor]
        #
        data_adjacency_matrix = data_adjacency_matrix.tocsr()
        self.data_adjacency_matrix = data_adjacency_matrix
        print self.data_adjacency_matrix.nnz
        #
        print 'saving the sparse matrix'
        sssm.save_sparse_csr('./data_adjacency_matrix', data_adjacency_matrix)
        #
        print 'saving mat format matrix for use in matlab'
        si.savemat('./data_adjacency_matrix.mat', dict(data_adjacency_matrix=data_adjacency_matrix))

    def plot_data_matrix(self):
        plt.close()
        plt.spy(p_obj.data_adjacency_matrix.todense())
        plt.savefig('./data_adjacency_matrix.png', dpi=300, format='png')
        plt.close()

    def prepare_data(self):
        data_arr_map = [None]*7
        data_arr_vocab_map = [None]*7
        #
        data_path0 = './Data for sparse word vectors/War and Peace.txt'
        data_arr_map[0] = self.read_data(data_path0)
        data_path0 = None
        data_arr_vocab_map[0] = self.get_vocab(data_arr_map[0])
        #
        data_path1 = './Data for sparse word vectors/Dibyesh - Hindu Nationalism Book.txt'
        data_arr_map[1] = self.read_data(data_path1)
        data_path1 = None
        data_arr_vocab_map[1] = self.get_vocab(data_arr_map[1])
        #
        data_path2 = './Data for sparse word vectors/Pattern recognition.txt'
        data_arr_map[2] = self.read_data(data_path2)
        data_path2 = None
        data_arr_vocab_map[2] = self.get_vocab(data_arr_map[2])
        #
        data_path3 = './Data for sparse word vectors/Elements of Information Theory.txt'
        data_arr_map[3] = self.read_data(data_path3)
        data_path3 = None
        data_arr_vocab_map[3] = self.get_vocab(data_arr_map[3])
        #
        data_path4 = './Data for sparse word vectors/Neurogenesis in the Adult Brain I.txt'
        data_arr_map[4] = self.read_data(data_path4)
        data_path4 = None
        data_arr_vocab_map[4] = self.get_vocab(data_arr_map[4])
        #
        data_path5 = './Data for sparse word vectors/The return of depression economics.txt'
        data_arr_map[5] = self.read_data(data_path5)
        data_path5 = None
        data_arr_vocab_map[5] = self.get_vocab(data_arr_map[5])
        #
        data_path6 = './Data for sparse word vectors/A country is not a company.txt'
        data_arr_map[6] = self.read_data(data_path6)
        data_arr_vocab_map[6] = self.get_vocab(data_arr_map[6])
        data_path6 = None
        #
        #
        # # find vocab common to any of the pairs
        # num_data_sets = len(data_arr_map)
        # assert num_data_sets == len(data_arr_vocab_map)
        # non_content_words = data_arr_vocab_map[0]
        # assert non_content_words is not None
        # for curr_idx_i in range(1, num_data_sets):
        #     assert data_arr_vocab_map[curr_idx_i] is not None
        #     non_content_words = np.intersect1d(non_content_words, data_arr_vocab_map[curr_idx_i], assume_unique=True)
        # non_content_words = np.unique(non_content_words)
        #
        #
        common_vocab = np.concatenate((data_arr_map[0], data_arr_map[1], data_arr_map[5], data_arr_map[6]))
        common_vocab = np.unique(common_vocab)
        #
        #
        data_arr_map[0:2] = [None]*2
        data_arr_vocab_map[5:] = [None]*2
        #
        data_arr_map[0:2] = [None]*2
        data_arr_vocab_map[5:] = [None]*2
        #
        # common_vocab = np.array([])
        # for curr_idx_i in range(num_data_sets):
        #     for curr_idx_j in range(num_data_sets):
        #         if curr_idx_i == curr_idx_j:
        #             continue
        #         else:
        #             curr_vocab_i = data_arr_vocab_map[curr_idx_i]
        #             curr_vocab_j = data_arr_vocab_map[curr_idx_j]
        #             #
        #             if (curr_vocab_i is None) or (curr_vocab_j is None):
        #                 continue
        #             #
        #             curr_ij_common_vocab = np.intersect1d(curr_vocab_i, curr_vocab_j, assume_unique=True)
        #             common_vocab = np.concatenate((common_vocab, curr_ij_common_vocab))
        # common_vocab = np.unique(common_vocab)
        # self.common_vocab = common_vocab
        #
        # self.common_vocab = non_content_words
        # common_vocab = None
        # print 'common_vocab', self.common_vocab.tolist()
        # print 'common_vocab.shape', self.common_vocab.shape
        #
        #
        #
        #
        data_arr = np.array([])
        for curr_idx in range(len(data_arr_map)):
            if data_arr_map[curr_idx] is not None:
                data_arr = np.concatenate((data_arr, data_arr_map[curr_idx]))
        self.data_arr = data_arr
        data_arr = None
        print 'self.data_arr', self.data_arr
        print 'self.data_arr', self.data_arr.shape
        data_arr_map = None
        self.vocab = self.get_vocab(data_arr=self.data_arr)
        #
        #
        common_vocab = np.intersect1d(common_vocab, self.vocab, assume_unique=True)
        self.common_vocab = common_vocab
        common_vocab = None
        print 'common_vocab', self.common_vocab.tolist()
        print 'common_vocab.shape', self.common_vocab.shape
        #
        #
        common_vocab_idx = []
        for curr_idx in range(self.vocab.size):
            if self.vocab[curr_idx] in self.common_vocab:
                common_vocab_idx.append(curr_idx)
        not_common_vocab_idx = np.setdiff1d(np.arange(self.vocab.size), common_vocab_idx, assume_unique=True)
        common_vocab_idx = None
        #
        self.update_data_stream_wd_vocab_idx()
        print 'p_obj.data_idx_in_vocab.tolist()', self.data_idx_in_vocab.tolist()
        self.compute_data_matrix()
        print 'p_obj.data_adjacency_matrix', self.data_adjacency_matrix
        #
        data_adjacency_matrix_selected = self.data_adjacency_matrix[not_common_vocab_idx, :]
        data_adjacency_matrix_selected = data_adjacency_matrix_selected.tocsc()
        data_adjacency_matrix_selected = data_adjacency_matrix_selected[:, not_common_vocab_idx]
        data_adjacency_matrix_selected = data_adjacency_matrix_selected.tocsr()
        print 'saving selected mat format matrix for use in matlab'
        si.savemat('./data_adjacency_matrix_selected.mat', dict(data_adjacency_matrix_selected=data_adjacency_matrix_selected))

if __name__ == '__main__':
    is_load = False
    #
    if is_load:
        p_obj = load_object()
    else:
        p_obj = ProcessTextFileDataFrWordVec()
        p_obj.prepare_data()
        p_obj.save_object()
    #
    p_obj.plot_data_matrix()

