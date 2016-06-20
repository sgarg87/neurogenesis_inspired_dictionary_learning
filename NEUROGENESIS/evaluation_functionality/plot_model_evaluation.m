
function plot_model_evaluation(model, dir_path)
    plot_correlation(model, dir_path);
    plot_learned_dictionary_size(model, dir_path);
end

function [learned_dictionary_sizes, correlation, error] = post_process_results(evaluation, D, dictionary_sizes, params)
    learned_dictionary_sizes = [];
    correlation = [];
    error = [];
    % 
    curr_idx = 0;
    for curr_dict_size = dictionary_sizes
        curr_idx = curr_idx + 1;
        % 
        if params.is_nonzero_dict_element_in_learned_size
            [~,nonzero_ind] = find(sum(abs(D{curr_dict_size})));
            curr_learned_dict_size = length(nonzero_ind);
            clear nonzero_ind;
        else
            curr_learned_dict_size = size(D{curr_dict_size}, 2);
        end
        %     
        learned_dictionary_sizes = [learned_dictionary_sizes; curr_learned_dict_size];
        %
        curr_evaluation = evaluation{curr_dict_size};
        % 
        curr_correlation = curr_evaluation.correlation;
        curr_correlation = curr_correlation(2, :);
        % 
        curr_error =  curr_evaluation.error;
        clear curr_evaluation;
        %
        error(curr_idx, :) = curr_error;
        correlation(curr_idx, :) = curr_correlation;
    end
end

function plot_correlation(model, dir_path)
    params = model.params;
    % 
    close;
    hold on;
    %
    legend_list = {};
    count = 0;
    %     
    if model.algorithms.random
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random, params);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'k--');
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'bx-');
        count = count + 1;
        legend_list{count} = 'neurogen-groupMairal';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'bo-');
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'rs-');       
        count = count + 1;
        legend_list{count} = 'groupMairal';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'gv-');              
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'md-');
        count = count + 1;
        legend_list{count} = 'Mairal';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        errorbar(learned_dictionary_sizes, mean(correlation'),std(correlation'),'c+-');       
        count = count + 1;
        legend_list{count} = 'neurogen-Mairal';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    xlabel('final dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
    ylim([0,1]);    
    curr_path = strcat(dir_path, sprintf('Figures/correlation_n%d_nz%d_T%d_new%d%s',params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

function plot_learned_dictionary_size(model, dir_path)
    params = model.params;
    % 
    close;
    hold on;
    %
    legend_list = {};
    count = 0;
    % 
    if model.algorithms.random
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random, params);
        plot(model.dictionary_sizes.random, learned_dictionary_sizes,'k--');
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        plot(model.dictionary_sizes.neurogen_group_mairal, learned_dictionary_sizes,'bx-');
        count = count + 1;
        legend_list{count} = 'neurogen-groupMairal';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        plot(model.dictionary_sizes.neurogen_sg, learned_dictionary_sizes,'bo-');
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        plot(model.dictionary_sizes.group_mairal, learned_dictionary_sizes,'rs-');
        count = count + 1;
        legend_list{count} = 'groupMairal';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        plot(model.dictionary_sizes.sg, learned_dictionary_sizes,'gv-');
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        plot(model.dictionary_sizes.mairal, learned_dictionary_sizes,'md-');
        count = count + 1;
        legend_list{count} = 'Mairal';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, correlation, error] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        plot(model.dictionary_sizes.neurogen_mairal, learned_dictionary_sizes,'c+-');
        count = count + 1;
        legend_list{count} = 'neurogen-Mairal';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    xlabel('initial dictionary size k');
    ylabel('learned dictionary size');
    curr_path = strcat(dir_path, sprintf('Figures/learnedk_n%d_nz%d_T%d_new%d%s',params.n,100*params.nonzero_frac,params.T,params.new_elements,params.adapt));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

