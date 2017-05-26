import math
import constants_paths as cp


def get_large_images(self, T, input_dim):
    n = math.sqrt(input_dim)
    assert ((n % 1) == 0)
    image_size = [n, n]
    #
    [flowers_data, animals_data, oxford_data] =\
        self.flower_building_images_online(T * 2, cp.dir_path, image_size)
    #
    num_data = flowers_data.shape[1]
    flowers_data_train = flowers_data[:, range(0, num_data, 2)]
    flowers_data_test = flowers_data[:, range(1, num_data, 2)]
    #
    num_data = animals_data.shape[1]
    animals_data_train = animals_data[:, range(0, num_data, 2)]
    animals_data_test = animals_data[:, range(1, num_data, 2)]
    #
    num_data = oxford_data.shape[1]
    oxford_data_train = oxford_data[:, range(0, num_data, 2)]
    oxford_data_test = oxford_data[:, range(1, num_data, 2)]
    #
    return oxford_data_train, oxford_data_test, flowers_data_train, flowers_data_test, animals_data_train, animals_data_test
