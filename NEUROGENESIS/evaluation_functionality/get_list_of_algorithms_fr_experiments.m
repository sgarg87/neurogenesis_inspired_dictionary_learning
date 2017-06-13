function algorithms = get_list_of_algorithms_fr_experiments()
    algorithms = struct();
    %     
    algorithms.mairal = false;
    algorithms.random = false;
    algorithms.group_mairal = false;
    algorithms.neurogen_group_mairal = false;
    algorithms.neurogen_mairal = false;
    algorithms.neurogen_sg = false;
    algorithms.sg = false;
    %     
    % setting it to false first and then true so that we can simply comment
    % the algorithms we don't want to use while running the experiments.     
    %
    algorithms.mairal = true;
%     algorithms.neurogen_mairal = true;
%     algorithms.group_mairal = true;
    algorithms.neurogen_group_mairal = true;
%     algorithms.random = true;
%     algorithms.neurogen_sg = true;
%     algorithms.sg = true;
end
