clear
clc
addpath('./libsvm-3.23/windows');

%% Load different datasets for various tasks
% name = '../HDA_IXMAS/cam_2_to_3.mat';
name = '../HDA_WIKI/wiki_img2txt.mat';
% name = '../HDA_Caltech_Office/W_S2D.mat';
paired_percent = 0.3;
[Xs,Ys,Xt,Yt,Xsp,Ysp,Xtp,Ytp] = load_data(name,paired_percent);

%% Set parameters
T = 5;
Dim = 100;
Alpha = [0.01,0.1,1,10,100];
Beta = [0.01,0.1,1,10,100];
Lamda = [0.01,0.1,1,10,100];
Gamma = [0.01,0.1,1,10,100];

n_parameters = length(Dim)*length(Alpha)*length(Beta)*length(Lamda)*length(Gamma);
parameters_cell = cell(n_parameters,1);
results_acc = zeros(n_parameters,6);
results_iter_acc = zeros(n_parameters,T);
results_iter_mmd = zeros(n_parameters,T);

i = 1;
for d=1:length(Dim)
    for alpha=1:length(Alpha)
        for beta=1:length(Beta)
            for lamda=1:length(Lamda)
                for gamma=1:length(Gamma)
                    options.dim = Dim(d);
                    options.alpha = Alpha(alpha);
                    options.beta = Beta(beta);
                    options.lamda = Lamda(lamda);
                    options.gamma = Gamma(gamma);
                    parameters_cell{i,1} = options;
                    i = i+1;
                end
            end
        end
    end
end

%% Optimize the algortihm
max_acc = 0;
for i=1:n_parameters
    options = parameters_cell{i,1};
    iter_acc = zeros(1,T);
    iter_mmd = zeros(1,T);
    [np,~] = size(Xsp);
    
    % generate pesudo labels for the test data in the target domain with
    % SVM classifier
    model = svmtrain(Ytp,Xtp,'-t 0 -q');
    [Yt_pseudo, svm_acc,~] = svmpredict(Yt((np+1):end),Xt(np+1:end,:), model, '-q');
    fprintf('initial acc with SVM: %06.4f\n', svm_acc(1));

    for iter=1:T
        [iter_acc(1,iter),iter_mmd(1,iter),Yt_pseudo] = ...
            JIP(Xs',Ys,Xt',Yt,Xsp',Ysp,Xtp',Ytp,Yt_pseudo,options);
        fprintf('iter: %01.0f  acc: %06.4f  mmd: %06.4f\n', ...
            iter, iter_acc(1,iter),iter_mmd(1,iter));
    end

    if iter_acc(1,T)>max_acc
        max_acc = iter_acc(1,T);
    end
    fprintf('param: %01.0f  acc_tar: %06.4f  max_acc: %06.4f\n', ...
        i,iter_acc(1,T),max_acc);
    results_acc(i,1:6) = [iter_acc(1,T),options.dim,options.alpha,...
        options.beta,options.lamda,options.gamma];
    results_iter_acc(i,:) = iter_acc;
    results_iter_mmd(i,:) = iter_mmd;
end

%% Save the results
results.acc = results_acc;
results.results_iter_acc = results_iter_acc;
results.results_iter_mmd = results_iter_mmd;
results.max_acc = max_acc;
results.paired_percent = paired_percent;
results.T = T;
results.Dim = Dim;
results.Alpha = Alpha;
results.Beta = Beta;
results.Lamda = Lamda;
results.Gamma = Gamma;

save_name = 'results';
save(save_name,'results');