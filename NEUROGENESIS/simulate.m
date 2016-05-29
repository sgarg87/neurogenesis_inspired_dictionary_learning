function [D,C,data] = simulate(data_type,n,k,T,nonzero_C,noise)
% generate random dictionary D, random sparse code C, and (noisy) data  D*C +noise

% data_type - type of data-generating distribution from the exp-family, e.g. Gaussian, Bernoulli, etc.
% n - dimensionality of the input (and output) of an autoencoder
% k - dictionary size  
% T - number of samples (time points/iterations in the online setting)

D = rand(n,k);  % random dictionary
D = D./sqrt(ones(n,1)*sum(D.^2));% normalize disctionary columns to  have 2-norm = 1


% generate a collection of random sparse codes C
C = rand(k,T);  

nonzero_frac = nonzero_C/k;

% now sparsify - set to 0 'sparsity' fraction of randomly selected entries
mask = zeros(k,T);
while nnz(find(mean(mask,1)==0)) || nnz(find(mean(mask,2)==0 ))  % all zero row or column
    mask = rand(k,T) > 1-nonzero_frac; % (P(0) = sparsity, P(1) = 1- sparsity for each entry)  

end
C = C.*mask;

 
% simulate 'noiseless' data (or, rather the natural parameters of the corresponding exp.family distributions)
THETA = D*C;

% now, simulate the data matrix  as a collection of i.i.d. random variables  of the given distribution type

switch data_type
	case 'Gaussian'
      	% assume that 'noise' is stdev, theta is mean
		data = normrnd(THETA,THETA*noise);
	case 'Bernoulli'
		% p(x=1) = e^theta / (1+e^theta) = 1/(1+e^-theta)
            P = 1./(1+exp(-THETA));
        	data = binornd(1,P);  
       	% data = P > 0.5;sparsity
	otherwise
end
if (find(mean(data) == 0))
    display 'all zero column in data';
    pause;
end
