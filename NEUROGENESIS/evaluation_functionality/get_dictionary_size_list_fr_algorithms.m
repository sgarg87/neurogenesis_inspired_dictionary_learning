function [dictionary_sizes] = get_dictionary_size_list_fr_algorithms(algorithms)
    dictionary_sizes = struct();
    %
    %     5 10 15 20 
    global_size = [5 10 15 20 25 50 100 150];% 200 250 350 500];
    %
    if algorithms.mairal
        dictionary_sizes.mairal = [global_size 200];
    end
    %
    if algorithms.random
        dictionary_sizes.random = global_size;
    end
    %
    if algorithms.group_mairal
        dictionary_sizes.group_mairal = [global_size 2000 3500];
    end
    %
    if algorithms.sg
        dictionary_sizes.sg = global_size;
    end
    %
    if algorithms.neurogen_mairal
        dictionary_sizes.neurogen_mairal = [5 10 15 20 25 50 75 100 150 200];
    end
    %
    if algorithms.neurogen_sg
        dictionary_sizes.neurogen_sg = global_size;
    end
    %
    if algorithms.neurogen_group_mairal
        dictionary_sizes.neurogen_group_mairal = global_size;
    end 
end
