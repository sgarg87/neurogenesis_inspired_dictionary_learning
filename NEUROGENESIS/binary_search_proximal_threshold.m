function threshold = binary_search_proximal_threshold(u, num_nonzero_dict_element, sparsity_margin)
%     fprintf('\nnum_nonzero_dict_element: %d. ', num_nonzero_dict_element);
    %
    assert(num_nonzero_dict_element < length(u));
    %
    abs_u = abs(u);
    %
    threshold = binary_search(abs_u, num_nonzero_dict_element, sparsity_margin);
end

% function top_k_search(abs_u, num_nonzero_dict_element)
%     error('not implemented');
%     %     
%     if num_nonzero_dict_element > 5
%         warning('very inefficient if num_nonzero_dict_element is high');
%     end
%     %
%     for curr_idx = 1:num_nonzero_dict_element
%         ~, max_abs_u_idx = max(abs_u);        
%     end
% end

function threshold = binary_search(abs_u, num_nonzero_dict_element, sparsity_margin)
    min_val = 0;
    max_val = max(abs_u);
    length_abs_u = length(abs_u);
    %
    count_iter = 0;
    %     
    while true
        %
        count_iter = count_iter + 1;
        if count_iter > max(length_abs_u, 1e4)
            pause;
%             error('running forever');
        end
        %         
        mean_val = (min_val+max_val)/2;
        num_mean = nnz(max(abs_u-mean_val, 0));
%         fprintf('%d, ', num_mean);
        %
        curr_val_diff = abs(max_val-min_val);
        if max_val ~= 0
            curr_val_diff = curr_val_diff/max_val;
        end
        if curr_val_diff < 1e-4
            threshold = mean_val;
            return;
        end
        clear curr_val_diff;
        %
        if abs(num_mean-num_nonzero_dict_element) <= sparsity_margin
            threshold = mean_val;
            return;
        elseif num_mean > num_nonzero_dict_element
            min_val = mean_val;
        elseif num_mean < num_nonzero_dict_element
            max_val = mean_val;
        else
            error('This code block should not be executed. something wrong in the conditions above.\n');
        end
    end
end
