
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
        e = errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'k--');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'bx-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'neurogen-groupMairal';
        legend_list{count} = 'NODL';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        e = errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'bo-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'rs-');       
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'groupMairal';
        legend_list{count} = 'NODL-';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        e = errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'gv-');              
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'md-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'Mairal';
        legend_list{count} = 'ODL';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, pearson_correlation, ~, ~] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(pearson_correlation'),std(pearson_correlation'),'c+-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'neurogen-Mairal';
        legend_list{count} = 'NODL+';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    set(gca, 'FontSize', 20);
    hline = findobj(gcf, 'type', 'line');
    set(hline,'LineWidth', 2);
    set(hline,'MarkerSize', 12);
%     set(findall(gca, 'Type', 'Line'),'LineWidth',2);
%     set(findall(gca, 'Type', 'Line'),'MarkerSize',12);
%     
    xlabel('Final dictionary size k');
    ylabel('Pearson correlation (true, predicted)');
%     
    xlim([0,inf]);
%     ylim([0,1]);
%     
    curr_path = strcat(dir_path, sprintf('Figures/%s_pearson_correlation_n%d_T%d_new%d_%s_%s__sparsecodes_%d__dictionarysparse_%d_%d', prefix, params.n,params.T,params.new_elements,params.adapt, params.data_set_name, floor(params.nonzero_frac*params.n), params.is_sparse_dictionary, floor(params.nz_in_dict*params.n)));
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
        e = errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'k--');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'bx-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'neurogen-groupMairal';
        legend_list{count} = 'NODL';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        e = errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'bo-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'rs-');       
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'groupMairal';
        legend_list{count} = 'NODL-';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        e = errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'gv-');              
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'md-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'Mairal';
        legend_list{count} = 'ODL';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, ~, spearman_correlation, ~] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(spearman_correlation'),std(spearman_correlation'),'c+-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'neurogen-Mairal';
        legend_list{count} = 'NODL+';
    end
    % 
    legend(legend_list);
    legend('location','SouthEast');
    %
    set(gca, 'FontSize', 20);
    hline = findobj(gcf, 'type', 'line');
    set(hline,'LineWidth', 2);
    set(hline,'MarkerSize', 12);
%     
    xlabel('Final dictionary size k');
    ylabel('Spearman correlation (true, predicted)');
    xlim([0,inf]);
%     ylim([0,1]);    
    curr_path = strcat(dir_path, sprintf('Figures/%s_spearman_correlation_n%d_T%d_new%d_%s_%s__sparsecodes_%d__dictionarysparse_%d_%d', prefix, params.n,params.T,params.new_elements,params.adapt, params.data_set_name, floor(params.nonzero_frac*params.n), params.is_sparse_dictionary, floor(params.nz_in_dict*params.n)));
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
        e = errorbar(learned_dictionary_sizes, mean(error'),std(error'),'k--');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(error'),std(error'),'bx-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'neurogen-groupMairal';
        legend_list{count} = 'NODL';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        e = errorbar(learned_dictionary_sizes, mean(error'),std(error'),'bo-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(error'),std(error'),'rs-');       
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'groupMairal';
        legend_list{count} = 'NODL-';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        e = errorbar(learned_dictionary_sizes, mean(error'),std(error'),'gv-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(error'),std(error'),'md-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'Mairal';
        legend_list{count} = 'ODL';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, ~, ~, error] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        e = errorbar(learned_dictionary_sizes, mean(error'),std(error'),'c+-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'neurogen-Mairal';
        legend_list{count} = 'NODL+';
    end
    % 
    legend(legend_list);
    legend('location','NorthEast');
    %
    set(gca, 'FontSize', 20);
    hline = findobj(gcf, 'type', 'line');
    set(hline,'LineWidth', 2);
    set(hline,'MarkerSize', 12);
%     
    xlim([0,inf]);
%     
    xlabel('Final dictionary size k');
    ylabel('Error (true, predicted)');
    curr_path = strcat(dir_path, sprintf('Figures/%s_error_n%d_T%d_new%d_%s_%s__sparsecodes_%d__dictionarysparse_%d_%d', prefix, params.n,params.T,params.new_elements,params.adapt, params.data_set_name, floor(params.nonzero_frac*params.n), params.is_sparse_dictionary, floor(params.nz_in_dict*params.n)));
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
        e = plot(model.dictionary_sizes.random, learned_dictionary_sizes,'k--');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'random-D';
    end
    % 
    if model.algorithms.neurogen_group_mairal
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.neurogen_group_mairal, model.neurogen_group_mairal.D, model.dictionary_sizes.neurogen_group_mairal, params);
        e = plot(model.dictionary_sizes.neurogen_group_mairal, learned_dictionary_sizes,'bx-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'neurogen-groupMairal';
        legend_list{count} = 'NODL';
    end
    % 
    if model.algorithms.neurogen_sg
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.neurogen_sg, model.neurogen_sg.D, model.dictionary_sizes.neurogen_sg, params);
        e = plot(model.dictionary_sizes.neurogen_sg, learned_dictionary_sizes,'bo-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'neurogen-SG';
    end
    % 
    if model.algorithms.group_mairal
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.group_mairal, model.group_mairal.D, model.dictionary_sizes.group_mairal, params);
        e = plot(model.dictionary_sizes.group_mairal, learned_dictionary_sizes,'rs-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'groupMairal';
        legend_list{count} = 'NODL-';
    end
    % 
    if model.algorithms.sg
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.sg, model.sg.D, model.dictionary_sizes.sg, params);
        e = plot(model.dictionary_sizes.sg, learned_dictionary_sizes,'gv-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
        legend_list{count} = 'SG';
    end
    % 
    if model.algorithms.mairal
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.mairal, model.mairal.D, model.dictionary_sizes.mairal, params);
        e = plot(model.dictionary_sizes.mairal, learned_dictionary_sizes,'md-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'Mairal';
        legend_list{count} = 'ODL';
    end
    % 
    if model.algorithms.neurogen_mairal
        [learned_dictionary_sizes, ~, ~, ~] = post_process_results(model.evaluation.neurogen_mairal, model.neurogen_mairal.D, model.dictionary_sizes.neurogen_mairal, params);
        e = plot(model.dictionary_sizes.neurogen_mairal, learned_dictionary_sizes,'c+-');
        e.LineWidth = 2;
        e.MarkerSize = 12;
        count = count + 1;
%         legend_list{count} = 'neurogen-Mairal';
        legend_list{count} = 'NODL+';
    end
    % 
    legend(legend_list);
    legend('location','NorthWest');
    %
    set(gca, 'FontSize', 20);
    hline = findobj(gcf, 'type', 'line');
    set(hline,'LineWidth', 2);
    set(hline,'MarkerSize', 12);
%     
    xlabel('Initial dictionary size k');
    ylabel('Learned dictionary size');
    curr_path = strcat(dir_path, sprintf('Figures/%s_learnedk_n%d_T%d_new%d_%s_%s__sparsecodes_%d__dictionarysparse_%d_%d', prefix, params.n,params.T,params.new_elements,params.adapt, params.data_set_name, floor(params.nonzero_frac*params.n), params.is_sparse_dictionary, floor(params.nz_in_dict*params.n)));
    saveas(gcf,curr_path,'fig');
    saveas(gcf,curr_path,'png');
    close(gcf);
end

