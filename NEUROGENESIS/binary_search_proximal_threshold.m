function threshold = binary_search_proximal_threshold(u, num_nonzero_dict_element, sparsity_margin)
%     fprintf('\nnum_nonzero_dict_element: %d. ', num_nonzero_dict_element);
    %
    assert(num_nonzero_dict_element < length(u));
    %
    min_val = 0;
    abs_u = abs(u);
    max_val = max(abs_u);
    %
%     count = 0;
    while true
        % 
        mean_val = (min_val+max_val)/2;
        num_mean = nnz(max(abs_u-mean_val, 0));
%         fprintf('%d, ', num_mean);
        %
        if abs(max_val-min_val) < 1e-4
            threshold = mean_val;
            return;
        end
%         count = count + 1;
%         if count > 100
%             threshold = mean_val;
%             assert(abs(num_mean-num_nonzero_dict_element) <= sparsity_margin*2);
%             return;            
%         end
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
