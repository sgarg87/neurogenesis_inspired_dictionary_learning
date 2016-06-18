#! /bin/bash

source /usr/usc/matlab/default/setup.sh
#
which matlab
#
matlab -nosplash -nodesktop -r "run('/auto/rcf-proj2/gv/sahilgar/sparse_dictionary_learning/code/neurogenesis_irina_rish/NEUROGENESIS/script_evaluate_online(true)'); exit"

