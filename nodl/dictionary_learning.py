import numpy as np
import math
import numpy.linalg as npl
import numpy.random as npr


class DictionaryLearning:
    #
    # todo: as per the settings of dictionary sparsity, change the format of dictionary from non-sparse to sparse (the initialized dictionary is non-sparse though).
    # todo: for high number of elements in a dictionary, see if we can have A and B matrix to be sparse. B can be sparse if data is sparse (and optionally codes are sparse). Whereas, A could be sparse if codes are sparse
    #
    def __init__(self,
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
                 dict_update_method,
                 batch_size,
                 num_samples_fr_processing,
                 is_grand_mother_neurons
    ):
        # dictionary
        self.D = None
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
        self.dict_update_method = dict_update_method
        #
        self.batch_size = batch_size
        #
        self.num_samples_fr_processing = num_samples_fr_processing
        #
        self.is_grand_mother_neurons = is_grand_mother_neurons

    def __initialize_dictionary__(self, n, k):
        self.D = npr.rand(n, k)

    def __initialize_memory__(self, n, k):
        self.A = np.zeros(shape=(k, k))
        self.B = np.zeros(shape=(n, k))

    def __update_dictionary__(self, max_num_iter, num_dict_elements, is_group_sparsity=False):
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
                        ceoff = max((1-(self.lambda_g/u_norm)), 0)
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
                num_dict_elements=num_dict_elements,
                is_group_sparsity=False
            )
        elif self.dict_update_method == 'group_mairal':
            self.__update_dictionary__(
                max_num_iter=max_num_iter,
                num_dict_elements=num_dict_elements,
                is_group_sparsity=True
            )
        else:
            raise NotImplementedError, 'no such dictionary update method implemented.'

    def dictionary_learning(self, x):
        print 'The initial number of dictionary elements is {}.'.format(self.D.shape[1])
        #
        n = self.D.shape[0]
        k = self.D.shape[1]
        num_data = x.shape[1]
        assert x.shape[0] == n
        #
        C = np.zeros(shape=(k, num_data))
        #
        if (self.A is None) or (self.B is None):
            assert (self.A is None) and (self.B is None)
            self.__initialize_memory__(n, k)
        #
        iter_start = 0
        iter_end = self.batch_size
        curr_iter = 0
        #
        while iter_end < self.num_samples_fr_processing:
            curr_iter += 1
            curr_data_batch = x[:, iter_start:iter_end]
            #
            # sparse coding
            curr_codes_batch = self.compute_codes(curr_data_batch)
            # todo: compute error and correlation metrics here
            #
            if self.max_new_elements > 0:
                if self.is_neurogenesis_conditional:
                    if curr_iter > 0:
                        # todo: computer birth rate based on error
                        raise NotImplemented
                    else:
                        birth_rate = 0
                    #
                    curr_new_elements_count = int(math.floor(self.max_new_elements*birth_rate))
                else:
                    curr_new_elements_count = self.max_new_elements
                #
                if curr_new_elements_count > 0:
                    #
                    print 'Adding new dictionary elements {}'.format(curr_new_elements_count)
                    #
                    new_dictionary_elements = npr.rand(n, curr_new_elements_count)
                    new_dictionary_elements = self.normalize_dictionary_elements(new_dictionary_elements)
                    #
                    if self.is_grand_mother_neurons:
                        raise NotImplemented
                    #
                    self.D = np.hstack((self.D, new_dictionary_elements))
                    #
                    # extend memory
                    self.B = np.hstack((self.B, np.zeros(shape=(n, curr_new_elements_count))))
                    self.A = np.hstack((self.A, np.zeros(shape=(k, curr_new_elements_count))))
                    self.A = np.vstack((self.A, np.zeros(shape=(curr_new_elements_count, k+curr_new_elements_count))))
                    #
                    # extend codes
                    C = np.vstack((C, np.zeros(shape=(curr_new_elements_count, C.shape[1]))))
                    #
                    k += curr_new_elements_count
                    #
                    curr_codes_batch = self.compute_codes(curr_data_batch)
                    curr_data_batch = None
            #
            self.A = self.A + np.outer(curr_codes_batch, curr_codes_batch.T)
            self.B = self.B +np.outer(curr_data_batch, curr_codes_batch.T)
            #
            C[:, iter_start:iter_end] = curr_codes_batch
            curr_codes_batch = None
            #
            self.update_dictionary()
            #
            # todo: add code related to deletion of zero dictionary elements if group sparsity constraint
            #
            iter_start = iter_end
            iter_end = iter_end + self.batch_size

    def __sparsify_dictionary_element__(self, dict_element):
        #
        # in python code, lars should be fast enough and accurate
        #
        if self.dictionary_element_sparse_alg == 'lars':
            raise NotImplementedError
        elif self.dictionary_element_sparse_alg == 'proximal':
            n = dict_element.size
            num_nonzero_dict_element = math.floor(self.dict_sparse_nnz_ratio * n)
            #
            if (num_nonzero_dict_element < n):
                #
                dict_element_lam = self.binary_search_proximal_threshold(
                        dict_element,
                        num_nonzero_dict_element,
                        max(0.01*num_nonzero_dict_element, 1)
                )
                #
                num_nonzero_dict_element = None
                #
                dict_element = self.proximal_operator(sol=dict_element, sparsity_lambda=dict_element_lam)
                dict_element_lam = None
            #
            return dict_element
        else:
            raise AssertionError

    def normalize_dictionary_elements(self, D):
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
            D = self.normalize_dictionary_elements(D)
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
                sol = self.proximal_operator(sol, sparsity_lambda=sparse_coding_lambda)
            #
            codes[:, curr_data_idx] = sol
        #
        # add code for computing error and pearson correlations, here or as a separate module, as a function of codes and dictionary
        #
        return codes

    def proximal_operator(self, sol, sparsity_lambda):
        sol_abs = np.abs(sol) - sparsity_lambda
        sol_abs[np.where(sol_abs < 0)] = 0
        sol_sign = np.sign(sol)
        sol = sol_sign * sol_abs
        sol_abs = None
        sol_sign = None
        return sol

    def binary_search_proximal_threshold(self, u, num_nonzeros, sparsity_margin):
        assert num_nonzeros < u.size
        lam_threshold = self.binary_search(np.abs(u), num_nonzeros, sparsity_margin)
        return lam_threshold

    def binary_search(self, abs_u, num_nonzero_dict_element, sparsity_margin):
        min_val = 0.0
        max_val = max(abs_u)
        n = abs_u.size
        #
        count_iter = 0
        #
        while True:
            count_iter += 1
            #
            if count_iter > max(n, 1e4):
                print 'warning: running forever'
            #
            mean_val = (min_val + max_val)/2
            #
            num_mean = np.count_nonzero(self.proximal_operator(abs_u, mean_val))
            #
            curr_val_diff = abs(max_val - min_val)
            #
            if max_val != 0:
                curr_val_diff /= max_val
            #
            if curr_val_diff < 1e-4:
                return mean_val
            #
            if abs(num_mean - num_nonzero_dict_element) <= sparsity_margin:
                return mean_val
            elif num_mean > num_nonzero_dict_element:
                min_val = mean_val
            elif num_mean < num_nonzero_dict_element:
                max_val = mean_val
            else:
                raise AssertionError
        #
        raise AssertionError

    def update_memory(self, alpha, x):
        # x is a batch of data
        # alpha is code for the data
        self.B += np.outer(x, alpha)
        self.A += np.outer(alpha, alpha)

