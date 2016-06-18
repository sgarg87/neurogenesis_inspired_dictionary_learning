function mu = link_func(theta,data_type)

% inverse link function for GLMs

switch data_type
	case 'Gaussian'
		mu = theta; % identity function
	otherwise
		mu = theta;

end