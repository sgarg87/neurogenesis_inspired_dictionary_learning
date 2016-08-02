
function plot_model_evaluation(model, dir_path, prefix)
    plot_pearson_correlation(model, dir_path, prefix);
    plot_spearman_correlation(model, dir_path, prefix);
    plot_error(model, dir_path, prefix);
    plot_learned_dictionary_size(model, dir_path, prefix);
end

function [learned_dictionary_sizes, pearson_correlation, spearman_correlation, error] = post_process_results(evaluation, D, dictionary_sizes, params)
    learned_dictionary_sizes = [];    
    spearman_correlation = [];
    pearson_correlation = [];
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
        pearson_correlation(curr_idx, :) = curr_evaluation.correlation(2, :);
        spearman_correlation(curr_idx, :) = curr_evaluation.correlation(1, :);
        error(curr_idx, :) = curr_evaluation.error;
        % 
        clear curr_evaluation;
    end
end

function plot_pearson_correlation(model, dir_path, prefix)
    params = model.params;
    % 
    close;
    hold on;
    %
    legend_list = {};
    count = 0;
    %     
    if model.algorithms.random
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random, params);
        errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'k--');
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'bx-');
        count = count + 1;
        legend_list{count} = 'neurogen-groupMairal';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'bo-');
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'rs-');       
        count = count + 1;
        legend_list{count} = 'groupMairal';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'gv-');              
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'md-');
        count = count + 1;
        legend_list{count} = 'Mairal';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'c+-');
        count = count + 1;
        legend_list{count} = 'neurogen-Mairal';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    xlabel('final dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
%     ylim([0,1]);    
    curr_path = strcat(dir_path, sprintf('Figures/%s_pearson_correlation_n%d_T%d_new%d_%s_%s', prefix, params.n,params.T,params.new_elements,params.adapt, params.data_set_name));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

function plot_spearman_correlation(model, dir_path, prefix)
    params = model.params;
    % 
    close;
    hold on;
    %
    legend_list = {};
    count = 0;
    %     
    if model.algorithms.random
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random, params);
        errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'k--');
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'bx-');
        count = count + 1;
        legend_list{count} = 'neurogen-groupMairal';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'bo-');
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'rs-');       
        count = count + 1;
        legend_list{count} = 'groupMairal';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'gv-');              
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'md-');
        count = count + 1;
        legend_list{count} = 'Mairal';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'c+-');
        count = count + 1;
        legend_list{count} = 'neurogen-Mairal';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    xlabel('final dictionary size k');
    ylabel('Spearman correlation (true, predicted)');
%     ylim([0,1]);    
    curr_path = strcat(dir_path, sprintf('Figures/%s_spearman_correlation_n%d_T%d_new%d_%s_%s', prefix, params.n,params.T,params.new_elements,params.adapt, params.data_set_name));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

function plot_error(model, dir_path, prefix)
    params = model.params;
    % 
    close;
    hold on;
    %
    legend_list = {};
    count = 0;
    %     
    if model.algorithms.random
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random, params);
        errorbar(learned_dictionary_sizes, mean(error'),std(error'),'k--');
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        errorbar(learned_dictionary_sizes, mean(error'),std(error'),'bx-');
        count = count + 1;
        legend_list{count} = 'neurogen-groupMairal';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        errorbar(learned_dictionary_sizes, mean(error'),std(error'),'bo-');
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        errorbar(learned_dictionary_sizes, mean(error'),std(error'),'rs-');       
        count = count + 1;
        legend_list{count} = 'groupMairal';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        errorbar(learned_dictionary_sizes, mean(error'),std(error'),'gv-');
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        errorbar(learned_dictionary_sizes, mean(error'),std(error'),'md-');
        count = count + 1;
        legend_list{count} = 'Mairal';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        errorbar(learned_dictionary_sizes, mean(error'),std(error'),'c+-');
        count = count + 1;
        legend_list{count} = 'neurogen-Mairal';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    xlabel('final dictionary size k');
    ylabel('Error (true, predicted)');
    curr_path = strcat(dir_path, sprintf('Figures/%s_error_n%d_T%d_new%d_%s_%s', prefix, params.n,params.T,params.new_elements,params.adapt, params.data_set_name));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

function plot_learned_dictionary_size(model, dir_path, prefix)
    params = model.params;
    % 
    close;
    hold on;
    %
    legend_list = {};
    count = 0;
    % 
    if model.algorithms.random
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.random, model.random.D, model.dictionary_sizes.random, params);
        plot(model.dictionary_sizes.random, learned_dictionary_sizes,'k--');
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        plot(model.dictionary_sizes.neurogen_group_mairal, learned_dictionary_sizes,'bx-');
        count = count + 1;
        legend_list{count} = 'neurogen-groupMairal';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        plot(model.dictionary_sizes.neurogen_sg, learned_dictionary_sizes,'bo-');
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        plot(model.dictionary_sizes.group_mairal, learned_dictionary_sizes,'rs-');
        count = count + 1;
        legend_list{count} = 'groupMairal';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        plot(model.dictionary_sizes.sg, learned_dictionary_sizes,'gv-');
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        plot(model.dictionary_sizes.mairal, learned_dictionary_sizes,'md-');
        count = count + 1;
        legend_list{count} = 'Mairal';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
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
    curr_path = strcat(dir_path, sprintf('Figures/%s_learnedk_n%d_T%d_new%d_%s_%s', prefix, params.n,params.T,params.new_elements,params.adapt, params.data_set_name));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

