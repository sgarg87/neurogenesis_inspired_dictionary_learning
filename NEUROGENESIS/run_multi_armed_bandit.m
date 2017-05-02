rng(0);
% kernel based hash codes for semantic paths
% 
% [model, ~] = multi_armed_bandit(false, 4, 25, false, true, true);
% save model_hashcodes model;
% % 
% [model, ~] = multi_armed_bandit(false, 4, 25, true, true, true);
% save model_hashcodes_dict model;
% 
% 
% ad click UCI data set
% 
% [model, ~] = multi_armed_bandit(false, 5, 10, false, true, true);
% save model_adclick model;
% 
[model, ~] = multi_armed_bandit(false, 5, 10, true, true, true);
save model_adclick_dict model;


