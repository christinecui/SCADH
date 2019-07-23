close all;
clear;

addpath(genpath('caffe/'));
addpath(genpath('utils/'))
% -- settings start here ---
% set 1 to use gpu, and 0 to use cpu
use_gpu = 1;

% top K returned images
top_k = 1000;
root_folder = './flickr_25';
feat_len = 32;
dataset = 'flickr_25';

% set result folder
result_folder = './analysis/32';

% models
model_file = './caffemodels/iter/DWDH_32bits_iter_6390.caffemodel';

% model definition
model_def_file = './caffemodels/vgg_feature32.prototxt';

% train-test
test_file_list = './flickr_25/test_file_list.txt';
test_label_file = './flickr_25/test_label.txt';
train_file_list = './flickr_25/retrival_file_list.txt';
train_label_file = './flickr_25/retrival_label.txt';

% --- settings end here ---

% caffe mode setting
phase = 'test'; % run with phase test (so that dropout isn't applied)

% --- settings end here ---

% outputs
feat_test_file = sprintf('%s/feat-test.mat', result_folder);
feat_train_file = sprintf('%s/feat-train.mat', result_folder);
binary_test_file = sprintf('%s/binary-test.mat', result_folder);
binary_train_file = sprintf('%s/binary-train.mat', result_folder);

% map and precision outputs
mean_th = 0;

% feature extraction- training set
if exist(feat_train_file, 'file') ~= 0  % file is exist
    load(feat_train_file);
    mean_bin = mean(feat_train');
    mean_th = mean(mean_bin);
    binary_train = (feat_train>mean_th);
else                                    % file is not exist
    feat_train = feat_batch(use_gpu, model_def_file, model_file, train_file_list, root_folder, feat_len);
    save(feat_train_file, 'feat_train', '-v7.3');
    mean_bin = mean(feat_train');
    mean_th = mean(mean_bin);
    binary_train = (feat_train>mean_th);
    save(binary_train_file,'binary_train','-v7.3');
end


% feature extraction- test set
if exist(feat_test_file, 'file') ~= 0
    load(feat_test_file);
    binary_test = (feat_test>mean_th);    
else
    feat_test = feat_batch(use_gpu, model_def_file, model_file, test_file_list, root_folder, feat_len);
    save(feat_test_file, 'feat_test', '-v7.3');
    binary_test = (feat_test>mean_th);
    save(binary_test_file,'binary_test','-v7.3');
end

% setting
trn_label = load(train_label_file);
tst_label = load(test_label_file);

B = compactbit(binary_train');
tB = compactbit(binary_test');

% add 20190605
% hash lookup: precision and reall 
hammRadius = 2;
traingnd   = trn_label;
testgnd    = tst_label;
cateTrainTest = bsxfun(@eq, traingnd, testgnd');
hammTrainTest = hammingDist(tB, B)';
Ret = (hammTrainTest <= hammRadius+0.00001);
[Pre, Rec] = evaluate_macro(cateTrainTest, Ret);

% add 20181018 start %
% output: map, Pre_5000
result_hrank = calcMapTopkMapTopkPreTopkRecLabel(tst_label, trn_label, tB, B, 5000);
map      = result_hrank.map;
Pre_5000 = result_hrank.topkPre;

% save
result_name = ['./analysis/32/' dataset '_map_Pre_topK_' datestr(now,30) '.mat'];
save(result_name, 'map', 'Pre_5000', 'Pre', 'Rec');
% add 20181018 end %