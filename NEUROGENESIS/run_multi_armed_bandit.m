% 
num_trials = 10;
% 
% 
% CNAE
% rng(0);
% [model, ~] = multi_armed_bandit(false, 1, num_trials, false, true, false);
% save model_cnae model;
% % 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 1, num_trials, true, true, false);
% save model_cnae_dict model;
% % 
% 
% % cover type
% rng(0);
% [model, ~] = multi_armed_bandit(false, 2, num_trials, false, true, false);
% save model_covertype model;
% 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 2, num_trials, true, true, false);
% save model_covertype_dict model;
% % 
% % 
% % Poker
% rng(0);
% [model, ~] = multi_armed_bandit(false, 6, num_trials, false, true, false);
% save model_poker model;
% % 
rng(0);
[model, ~] = multi_armed_bandit(false, 6, num_trials, true, true, false);
save model_poker_dict model;
% % 
% % 
% % Kernel based hash codes for semantic paths
% rng(0);
% [model, ~] = multi_armed_bandit(false, 4, num_trials, false, true, false);
% save model_hashcodes model;
% % 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 4, num_trials, true, true, false);
% save model_hashcodes_dict model;
% % 
% % 
% % Internet ad click UCI data set
% % 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 5, num_trials, false, true, false);
% save model_adclick model;
% % 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 5, num_trials, true, true, false);
% save model_adclick_dict model;
% % 
% % 
% % FAO images
% rng(0);
% [model, ~] = multi_armed_bandit(false, 8, num_trials, false, true, false);
% save model_fao model;
% % 
% rng(0);
% [model, ~] = multi_armed_bandit(false, 8, num_trials, true, true, false);
% save model_fao_dict model;
% 
% 
