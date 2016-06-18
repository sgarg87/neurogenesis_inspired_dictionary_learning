function [dictionary_sizes] = get_dictionary_size_list_fr_algorithms(algorithms)
    dictionary_sizes = struct();
    %
    global_size = [5 10 15 20 25 50 75 100 150 200 350 500 750 1000 1500 2000];
    %
    if algorithms.mairal
        dictionary_sizes.mairal = [5 10 15 20 25 50 75 100 150 200 350 500 1000];
    end
    %
    if algorithms.random
        dictionary_sizes.random = global_size;
    end
    %
    if algorithms.group_mairal
        dictionary_sizes.group_mairal = [5 10 15 20 25 50 75 100 150 200 350 500 750 1000 1500 2000 3500 5000];
    end
    %
    if algorithms.sg
        dictionary_sizes.sg = global_size;
    end
    %
    if algorithms.neurogen_mairal
        dictionary_sizes.neurogen_mairal = global_size;
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
