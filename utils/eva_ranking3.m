function [evaluation_info] = eva_ranking3(Dhamm, learn_lable, test_lable)

% Dhamm: learn * test
tmp_mat = test_lable*learn_lable'; % test  * learn
rel_mat = tmp_mat>=1;              % test  * learn


[~, sort_idx]= sort(Dhamm, 1, 'ascend'); % col--test
for i = 1:size(Dhamm, 2) % col--test
    database_label = rel_mat(i, :); %
    ret_label = database_label(sort_idx(:,i)); % size 1*22500

    for j = 500:500:5000
        precision(i, j/500) = length(find(ret_label(1:j)==1))/j ;
        recall(i, j/500)    = length(find(ret_label(1:j)==1))/(sum(ret_label)+eps);
    end    
end

P_TopK = mean(precision); 
R_TopK = mean(recall);    

MAP = mAPs(Dhamm, learn_lable, test_lable, 1);

%% 
evaluation_info.precision = P_TopK;
evaluation_info.recall    = R_TopK;
evaluation_info.mAP    = MAP;
