import numpy as np
import math
import numpy.linalg as npl


class DictionaryLearning:
    #
    # todo: as per the settings of dictionary sparsity, change the format of dictionary from non-sparse to sparse (the initialized dictionary is non-sparse though).
    # todo: for high number of elements in a dictionary, see if we can have A and B matrix to be sparse. B can be sparse if data is sparse (and optionally codes are sparse). Whereas, A could be sparse if codes are sparse
    #
    def __init__(self,
                 X,
                 D,
                 alg,
                 A,
                 B,
                 coding_sparse_algo,
                 code_sparse_nnz_ratio,
                 is_sparse_dictionary,
                 dictionary_element_sparse_alg,
                 dict_sparse_nnz_ratio,
                 is_sparse_data,
                 lambda_g,
                 max_new_elements,
                 epsilon,
                 gamma,
                 is_neurogenesis_conditional,
                 dict_update_method
    ):
        # dictionary
        self.D = D
        #
        # algorithm to learn the dictionary
        self.alg = alg
        #
        # memory matrices representing all the data and the corresponding codes in a compact manner
        self.A = A
        self.B = B
        #
        #
        # codes related sparsity settings (so called "sparse coding")
        self.coding_sparse_algo = coding_sparse_algo
        self.code_sparse_nnz_ratio = code_sparse_nnz_ratio
        #
        # dictionary elements related sparsity settings
        self.is_sparse_dictionary = is_sparse_dictionary
        self.dictionary_element_sparse_alg = dictionary_element_sparse_alg
        self.dict_sparse_nnz_ratio = dict_sparse_nnz_ratio
        #
        # if data is sparse, the evaluation metrics and the normalization conditions in the learning may differ.
        # Also, the sparsity of data can be exploited for efficient computations in the code
        self.is_sparse_data = is_sparse_data
        #
        # killing of elements
        self.lambda_g = lambda_g
        #
        # maximum number of elements for the processing of a batch (birth of neurons)
        self.max_new_elements = max_new_elements
        #
        # convergence threshold for updating of dictionary using the block coordinate decent
        self.epsilon = epsilon
        #
        # conditional neurogenesis related settings
        self.is_neurogenesis_conditional = is_neurogenesis_conditional
        # upper bound on Pearson correlation for conditional neurogenesis
        self.gamma = gamma
        #
        #
        self.dict_update_method = dict_update_method

    def __initialize_dictionary__(self):
        raise NotImplemented

    def __initialize_memory__(self):
        raise NotImplemented

    def __update_dictionary__(self, max_num_iter, dict_elements_dimension, num_dict_elements, is_group_sparsity=False):
        is_converged = False
        curr_count = 0
        #
        while not is_converged:
            curr_count += 1
            #
            if curr_count > max_num_iter:
                break
            #
            # todo: this could be costly, see if we can avoid i
            D_prev = np.copy(self.D)
            #
            # block coordinate decent procedure
            for curr_dict_element_idx in range(num_dict_elements):
                #
                # todo: is this step really necessary ?
                if is_group_sparsity:
                    #  all-zeros dictionary element - skip it
                    if np.count_nonzero(self.D[:, curr_dict_element_idx]) == 0:
                        continue
                #
                if self.A[curr_dict_element_idx, curr_dict_element_idx] == 0:
                    a = 1e-100
                else:
                    a = self.A[curr_dict_element_idx, curr_dict_element_idx]
                #
                u = \
                    (
                        self.B[:, curr_dict_element_idx]
                        -
                        self.D.dot(self.A[:, curr_dict_element_idx])
                    ) \
                    + \
                    (
                        self.A[curr_dict_element_idx, curr_dict_element_idx] * self.D[:, curr_dict_element_idx]
                    )
                #
                if not self.is_sparse_dictionary:
                    u = u / a
                else:
                    # todo: make sure this operation doesn't cost too much
                    if np.any(u != 0):
                        u = u / a
                        u = self.__sparsify_dictionary_element__(u)
                        assert np.isnan(u).sum() == 0
                #
                #
                if is_group_sparsity:
                    u_norm = float(math.sqrt(u.dot(u)))
                    if u_norm == 0:
                        coeff = 0
                    else:
                        ceoff = max((1-(self.lambda_D/u_norm)), 0)
                    u_norm = None
                    # soft thresholding
                    u = coeff*u
                #
                u_norm = float(math.sqrt(u.dot(u)))
                u = u*(1.0/max(1.0, u_norm))
                u_norm = None
                #
                self.D[:, curr_dict_element_idx] = u
                u = None
            #
            max_diff = np.abs(D_prev - self.D).max()
            print 'max_diff', max_diff
            #
            if max_diff < self.epsilon:
                is_converged = True

    def update_dictionary(self):
        #
        # todo: make the necessary changes for efficient computations with sparse dictionaries
        #
        assert len(self.D.shape) == 2
        dict_elements_dimension = self.D.shape[0]
        num_dict_elements = self.D.shape[1]
        #
        # max number of iterations for update
        max_num_iter = min(max(0.01*dict_elements_dimension, 5), 20)
        #
        if self.dict_update_method == 'mairal':
            self.__update_dictionary__(
                max_num_iter=max_num_iter,
                dict_elements_dimension=dict_elements_dimension,
                num_dict_elements=num_dict_elements,
                is_group_sparsity=False
            )
        elif self.dict_update_method == 'group_mairal':
            self.__update_dictionary__(
                max_num_iter=max_num_iter,
                dict_elements_dimension=dict_elements_dimension,
                num_dict_elements=num_dict_elements,
                is_group_sparsity=True
            )
        else:
            raise NotImplementedError, 'no such dictionary update method implemented.'

    def dictionary_learning(self):
        raise NotImplemented

    def __sparsify_dictionary_element__(self, dict_element):
        raise NotImplementedError

    def normalize_dictionary(self, D):
        raise NotImplemented

    def compute_codes(self, x):
        #
        # todo: make appropriate efficiency related changes for the case of sparse dictionary, using scipy linear algebra package
        #
        input_dimension = self.D.shape[0]
        k = self.D.shape[1]
        #
        code_sparse_nnz = math.floor(input_dimension*self.code_sparse_nnz_ratio)
        print 'code_sparse_nnz', code_sparse_nnz
        #
        num_data_points = x.shape[1]
        print 'num_data_points', num_data_points
        #
        assert code_sparse_nnz >= 0
        #
        D = np.copy(self.D)
        #
        # normalize the dictionary
        if not self.is_sparse_data:
            D = self.normalize_dictionary(D)
        #
        codes = np.zeros(shape=(k, num_data_points))
        #
        for curr_data_idx in range(num_data_points):
            curr_data = x[:, curr_data_idx]
            #
            assert self.coding_sparse_algo == 'proximal'
            #
            sol = npl.lstsq(D, curr_data)[0]
            if code_sparse_nnz < k:
                sparse_coding_lambda = self.binary_search_proximal_threshold(sol, code_sparse_nnz, max(0.01*code_sparse_nnz, 1))
                sol_abs = np.abs(sol)-sparse_coding_lambda
                sol_abs[np.where(sol_abs < 0)] = 0
                sol_sign = np.sign(sol)
                sol = sol_sign*sol_abs
                sol_abs = None
                sol_sign = None
            #
            codes[:, curr_data_idx] = sol
        #
        # add code for computing error and pearson correlations, here or as a separate module, as a function of codes and dictionary
        #
        return codes

    def binary_search_proximal_threshold(self):
        pass

    def update_memory(self, alpha, x):
        # x is a batch of data
        # alpha is code for the data
        pass

