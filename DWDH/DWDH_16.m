clear all;
nbit = 16;
fprintf('======start %d bits DWDH======\n\n', nbit);
    
%% set output save path
addpath(genpath('../utils/'));
dataset = 'flickr_25';
use_kmeans = 1;
folder_name = '../data_from_DWDH';
if ~exist(folder_name, 'dir')
    mkdir(folder_name)
end
save_path = sprintf('%s/B_%dbits.h5', folder_name, nbit);
fprintf('---------------------------------------\n');
fprintf('save path is %s\n', save_path);
fprintf('---------------------------------------\n');

%% load dataset
fprintf('load dataset %s...\n', dataset);
load('../fc7_features/traindata_16.txt');
load ../flickr_25/train_tag.txt;
F = train_tag;

traindata = double(traindata_16');
fprintf('Finished!\n');                                                         
fprintf('---------------------------------------\n'); 

%% data prepocess
fprintf('data prepocessing...\n');
X = traindata; 
X = normalize(X');
traindata = traindata';
sampleMean = mean(traindata, 1);                                                       
traindata = (traindata - repmat(sampleMean, size(traindata, 1), 1));
[n, d] = size(X);

c = size(F, 2);
fprintf('Finished!\n');
fprintf('---------------------------------------\n');

%% parameters set
nMaxIter = 100; 

% test 5
% n_anchors = 500;
% s = 3;
% mu = 0.1;
% pho=1.1; 
% lambda1 = 0.01; 
% lambda2 = 0.001; 
% lambda3 = 0.1;
% lambda6 = 100;

% test 6
n_anchors = 500;
s = 3;
mu = 1;
pho=1.1; 
lambda1 = 10; 
lambda2 = 100; 
lambda3 = 10;
lambda6 = 100;


sigma = 0;

if ~use_kmeans
     anchor = traindata(randsample(n, n_anchors),:);
else
    fprintf('K-means clustering to get m anchor points\n');
    %[~, anchor] = litekmeans(traindata, n_anchors, 'MaxIter', 30);
    [~,anchor] = litekmeans(traindata ,n_anchors,'MaxIter',5,'Replicates',1);
    fprintf('anchor points have been selected!\n');
    fprintf('---------------------------------------\n');
end

%% Generating anchor graphs
fprintf('Generating anchor graphs\n');
Z = zeros(n, n_anchors);
Dis = sqdist(traindata', anchor');
clear X;    
%clear traindata;
clear anchor;

val = zeros(n, s);
pos = val;
for i = 1:s
    [val(:,i), pos(:,i)] = min(Dis, [], 2);
    tep = (pos(:,i) - 1) * n + [1:n]';
    Dis(tep) = 1e60;
end
clear Dis;
clear tep;

if sigma == 0
    sigma = mean(val(:,s) .^ 0.5);
end
val = exp(-val / (1 / 1 * sigma ^ 2));
val = repmat(sum(val, 2).^ -1, 1, s) .* val;  % dim=2, row  normalization
tep = (pos - 1) * n + repmat([1:n]', 1, s);
Z([tep]) = [val];
clear tep;
clear val;
clear pos;
 
lamda = sum(Z);                % 1 * 500
lambda = diag(lamda .^ -1);    % 500 * 500
%size(lambda)                   % 500 * 500
%size(Z)                        % 118000 * 500
clear lamda;
fprintf('Finished!\n');
fprintf('---------------------------------------\n'); 

%% initization
B = sign(randn(n, nbit));     
Y = sign(randn(n, c));  
Y1 = Y;
Y2 = Y;
Gamma1 = zeros(size(F));   
Gamma2 = zeros(size(F));  
Gamma3 = zeros(size(B));               

%% loop
i = 0; 
loss_old = 0;

while i < nMaxIter
    i = i + 1;  

    %% W1 c*nbit
    tempW1 = lambda2*(Y'*Y) + lambda3*eye(c);  
    W1 = lambda2*(tempW1\(Y'*B)); 
    
%    %% W2 d*nbit
%    tempW2 = lambda4*(X'*X) + lambda5*eye(d);  
%    W2 = lambda4*(tempW2\(X'*B)); 

    %% update Y
    tempY = lambda2*(W1*W1') + 2*mu*eye(c);
    Y = (lambda2*B*W1' + mu*Y1 + mu*Y2 - Gamma1 - Gamma2)/tempY;

    %% update Y1
    [U, S, V] = svd(Y + (1/mu)*Gamma1, 'econ');
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    if svp>=1
        diagS = diagS(1:svp)-1/mu;
    else
        svp = 1;
        diagS = 0;
    end
    Y1 = U(:, 1:svp) * diag(diagS) * V(:, 1:svp)'; 

    %% update Y2    
    temp_T = Y - F + (1/mu)*Gamma2;
    E_hat = min(0, temp_T + lambda1/mu) + max(0, temp_T - lambda1/mu);
    Y2 = E_hat + F;
    
    %% C
    C = (1/mu) * (mu*B + Gamma3 -lambda6*B + lambda6*Z*lambda*(Z'*B));
    
    %% B
    %B = sign(lambda2*Y*W1 + lambda4*X*W2 - lambda6*C + lambda6*Z*lambda*(Z'*C) + mu*C - Gamma3);         
    B = sign(lambda2*Y*W1 - lambda6*C + lambda6*Z*lambda*(Z'*C) + mu*C - Gamma3);    
   
    %% 
    Gamma1 = Gamma1+ mu*(Y - Y1);
    Gamma2 = Gamma2+ mu*(Y - Y2);
    Gamma3 = Gamma3+ mu*(B - C);
    mu     = pho*mu; 
    
    % 1
    Temp1 = sum(svd(Y));             % ||Y||_*
    % 2
    Temp2 = sum(sum(abs(F - Y)));    % ||F - Y||_1
    Temp2 = lambda1 * Temp2;
    % 3
    Temp3 = B-Y*W1;     
    Temp3 = (lambda2 / 2) * trace(Temp3'*Temp3);
    % 4 
    Temp4 = (lambda3 / 2) * trace(W1'*W1);
%    % 5
%    Temp5 = B-X*W2;
%    Temp5 = (lambda4 / 2) *trace(Temp5'*Temp5);    
%    
%    % 6
%    Temp6 = (lambda5 / 2)*trace(W2'*W2);
    
    % 7
    Temp7 = lambda6*trace(B'*B-B'*Z*lambda*(Z'*B));
    
    % total  
    %loss = Temp1 + Temp2 + Temp3 + Temp4 + Temp5 + Temp6 + Temp7;
    loss = Temp1 + Temp2 + Temp3 + Temp4 + Temp7;

    res = (loss - loss_old)/loss_old;
    loss_old = loss;
    
    fprintf('iteration %3d, loss is %.4f, residual error is %.5f\n', i, loss, res);
    fprintf('---------------------------------------\n'); 
%     if (abs(res) <= 1e-4)
%         break;
%     end    
end

%fprintf('iteration %3d\n', i);
fprintf('======end %d bits DWDH======\n\n', nbit);

final_B = B;
final_B = sign(final_B);

fprintf('save B and final_B as HDF5 file\n');
fprintf('save path is %s\n' ,save_path);
h5create(save_path, '/final_B',[size(final_B, 2) size(final_B, 1)]);
h5create(save_path, '/B',[size(B, 2) size(B, 1)]);
h5write(save_path, '/final_B', final_B');
h5write(save_path, '/B', B');
fprintf('Finished!\n');                                                         
fprintf('---------------------------------------\n'); 
