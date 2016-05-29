function [D,err,correl_all] = DL(data,D0,nonzeros_C,lambda_D,mu,eta,epsilon,T,new_elements,data_type,D_update_method)
% learn a dictionary D, and a  sparse code C, for data

% parameters:
% nonzeros_C - regularization parameter for optimizing sparse code C: for LARS, # of nonzeros
%              if nonzeros = -1, the sparsity level is selected automatically as  best-fitting for each given sample
% lambda_D - regularization parameter for optimizing the dictionary (weight on group l1/l2); 
%			if 0, no dictionary elements are ever removed
% eta - learning rate in stochastic gradient descent 
% T  - number of iterations
% new_elements - the  number of new dictionary elements to generate per each sample (if 0, no neurogen occuring)
% data_type - 'Gaussian', or 'Bernoulli', or another exp-family


n = size(D0,1);
k = size(D0,2);

D = D0;
C = [];

A = zeros(k,k); B = zeros(n,k); % matrices used by Mairal's dictionary update method

[n1,n2]=size(data);
batch_size = 20;
t_start=1;
t_end=batch_size;
t = 0; % iteration

if (find(mean(data) == 0))
    display 'all zero column in data';
    pause;
end

while t_end <= T  % up to T samples
%     display 'iter'
      t = t+1;
	%  get the next batch of input samples 
      x = data(:,t_start:t_end);
      data_history = data(:,1:t_end);

      t_start=t_end+1;
      t_end = t_end + batch_size;

	% 1. neurogenesis step
    if  new_elements < 0   % just use the initial dictionary
		D = D0;
    end
    % changed on 11/9
     % evaluate the current dictionary before adding random elements 
    
    [code,err(t,:),correl] = sparse_coding(x,D,nonzeros_C,data_type);
    correl_S(t,:) = correl(1,:);
    correl_P(t,:) = correl(2,:);
    
    pre_err = err; pre_correl = correl; 
    pre_correl_P(t,:)=correl(1,:); pre_correl_S(t,:)=correl(2,:);
    
    
        
    if new_elements <0 % initial dictionary    
		continue;
    end
 

    % neurogen version  
      if  new_elements > 0  % include new dictionary columns
		D = [D normalize(rand(n,new_elements))]; 
        B = [B zeros(n,new_elements)];
        A = [A zeros(k,new_elements)];  
        A = [A;zeros(new_elements,k+new_elements)];
		k = k + new_elements;
	end

	% 2. sparse coding step AFTER adding random elements,to use it in dict update
    %   for each element in the data batch

      [code] = sparse_coding(x,D,nonzeros_C,data_type);

     % matrices used by Mairal's dictionary update method
     
      A = A + code*code';  
      B = B + x*code';
	
      if nnz(isnan(A))
        display 'A is NaN';
        pause;
      end
	% 3. dictionary learning step
	%    currently, implemented as 'truncated' stochastic gradient, or proximal method with group sparsity
    
    %%% just in case, giving all previous data and current encoding to updateD
    [code_history] = sparse_coding(data_history,D,nonzeros_C,data_type);

	[D,A,B] = updateD(D,code,x,lambda_D,mu,eta,epsilon,data_type,D_update_method, A,B, data_history,code_history) ;
   
    [nzD,ind] = find(sum(abs(D)));
    disp(num2str(length(ind)));
    if ~length(ind) 
        display 'empty dictionary!'
        pause;
    else
            D = D(:,ind);
            B = B(:,ind);
            A = A(:,ind); A = A(ind,:);
            k = length(ind);
    end
    
    [code,post_err(t,:),post_correl] = sparse_coding(x,D,nonzeros_C,data_type);
    
    post_correl_S(t,:) = post_correl(1,:);
    post_correl_P(t,:) = post_correl(2,:);
    
end

[er,ec] = size(err);
err = reshape(err',1,er*ec);
 
correl_all(1,:) = reshape(correl_P',1,er*ec);
correl_all(2,:) = reshape(correl_S',1,er*ec);
%err = mean(err')';
%correl = mean(correl')';