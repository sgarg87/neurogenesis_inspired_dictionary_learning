function dir_path = get_dir_path(is_hpcc)
	if is_hpcc
		% todo: correct this path, not complete.
		dir_path = '/auto/rcf-proj2/gv/sahilgar/sparse_dictionary_learning/code/neurogenesis_irina_rish/NEUROGENESIS/';
	else
		dir_path = './';
	end
end
