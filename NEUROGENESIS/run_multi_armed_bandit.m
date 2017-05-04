% 
% 
% kernel based hash codes for semantic paths
% 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 4, 25, false, true, true);
% save model_hashcodes model;
% % 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 4, 25, true, true, true);
% save model_hashcodes_dict model;
% 
% 
% internet ad click UCI data set
% 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 5, 10, false, true, false);
% save model_adclick model;
% % 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 5, 10, true, true, false);
% save model_adclick_dict model;
% 
% 
% CIFAR images
% 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 7, 10, false, true, false);
% save model_cifar model;
% 
rng(0);
[model, ~] = multi_armed_bandit(false, 7, 10, true, true, false);
save model_cifar_dict model;
%
% 
% 
% Our images
% 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 8, 10, false, true, false);
% save model_fao model;
% 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 8, 10, true, true, false);
% save model_fao_dict model;


