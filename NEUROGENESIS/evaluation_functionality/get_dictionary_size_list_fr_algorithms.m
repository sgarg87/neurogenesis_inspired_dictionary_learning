function [dictionary_sizes] = get_dictionary_size_list_fr_algorithms(algorithms)
    dictionary_sizes = struct();
    %
% 50 100 150 200 350 500
% 2 3 4
% [5 10 15 20 25 35 50 75 100 125 150 200 250]
% 
%  
% 
% 1500 2500 5000 10000
% 350 500 750 1000
% 100 150
% 
% 
    global_size = [25 50 100 150 250 350 500 750 1000];
% 
%     global_size = [25 100 250];
% 
%     global_size = [25 50 100 150 250 350 500];
% 
%     global_size = [25 100 500];
    %
%     
    if algorithms.mairal
        % 250 350 500 750 1000
        % 200
        dictionary_sizes.mairal = [global_size];
    end
    %
    if algorithms.random
        dictionary_sizes.random = dictionary_sizes.mairal;
    end
    %
    if algorithms.group_mairal
        dictionary_sizes.group_mairal = [global_size];
    end
    %
    if algorithms.sg
        dictionary_sizes.sg = global_size;
    end
    %
    if algorithms.neurogen_mairal
        dictionary_sizes.neurogen_mairal = [global_size];
    end
    %
    if algorithms.neurogen_sg
        dictionary_sizes.neurogen_sg = global_size;
    end
    %
    if algorithms.neurogen_group_mairal
        dictionary_sizes.neurogen_group_mairal = [ global_size ];
    end 
end
