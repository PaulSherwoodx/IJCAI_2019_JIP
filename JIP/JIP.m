function [acc_tar,mmd_new,pred_tar] = JIP(Xs,Ys,Xt,Yt,Xsp,Ysp,Xtp,Ytp,Yt_pseudo,options)
% The implementation of the proposed joint information preservation
% Xs: ds * ns
% Xt: dt * nt

dim = options.dim;
alpha = options.alpha; % for the local geometric information preservation
beta = options.beta; % for the paired information preservation
lamda = options.lamda; % for the global geometric information preservatio
gamma = options.gamma; % for the regularization term for well-defined problem

[ns,ds] = size(Xs');
[nt,dt] = size(Xt');
[~,np] = size(Xsp);

Yt_pseudo = [Ytp;Yt_pseudo];
X = [Xs,zeros(ds,nt);zeros(dt,ns),Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
Xs = X(1:ds,1:ns);
Xt = X(ds+1:end,ns+1:end);
Xsp = X(1:ds,1:np);
Xtp = X(ds+1:end,ns+1:ns+np);

%% construct MMD matrix for distribution matching
% to minimize Tr(W'X'MXP) for marginal distribution;
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e * e';

% to minimize Tr(W'X'McXP) for conditional distribution
number_class = length(unique(Ys));
N = 0;
if ~isempty(Yt_pseudo) && length(Yt_pseudo)==nt
    for c = reshape(unique(Ys),1,number_class)
        e = zeros(ns+nt,1);
        e(Ys==c) = 1 / length(find(Ys==c));
        e(ns+find(Yt_pseudo==c)) = -1 / length(find(Yt_pseudo==c));
        e(isinf(e)) = 0;
        N = N + e*e';
    end
end

% joint distribution adaptation
M = M + N;
M = M / norm(M,'fro');

%% construct correlation matrix for paired information preservation
H = eye(np)-ones(np)/np;
C = [zeros(ds,ds),Xsp*H*Xtp';Xtp*H*Xsp',zeros(dt,dt)];

%% construct Laplacian matrix for local geometric information preservation
manifold.k = 5;
manifold.NeighborMode = 'Supervised';
manifold.bNormalized = 1;
manifold.WeightMode = 'Cosine';
manifold.gnd = [Ys;Yt_pseudo];
Graph_W = constructW(X',manifold);
Graph_D = diag(full(sum(Graph_W,2)));
L = -Graph_W;
for i=1:size(L,1)
    L(i,i) = L(i,i) + Graph_D(i,i);
end

%% construct scatter matrix for global geometric information preservation
S_sb = zeros(ds,ds);
S_sw = zeros(ds,ds);
mean_s = mean(Xs,2);
mean_c_s = zeros(ds,number_class);

S_tb = zeros(dt,dt);
S_tw = zeros(dt,dt);
mean_t = mean(Xt,2);
mean_c_t = zeros(dt,number_class);

for c=1:number_class
    labels = unique(Ys);
    label = labels(c);
    
    % calculate between class scatter matrix and within class scatter
    % matrix for the data in the source domain
    index_s = find(Ys==label);
    num_c_s = length(index_s); 
    mean_c_s(:,c) = mean(Xs(:,index_s),2);
    H_c_s = eye(num_c_s)-ones(num_c_s,num_c_s)/num_c_s;
    
    S_sb = S_sb + num_c_s*(mean_c_s(:,c)-mean_s)*(mean_c_s(:,c)-mean_s)';
    S_sw = S_sw + Xs(:,index_s)*H_c_s*Xs(:,index_s)';
    
    % calculate between class scatter matrix and within class scatter
    % matrix for the data in the target domain
    index_t = find(Yt_pseudo==label);
    num_c_t = length(index_t);
    mean_c_t(:,c) = mean(Xt(:,index_t),2);
    H_c_t = eye(num_c_t)-ones(num_c_t,num_c_t)/num_c_t;
    
    S_tb = S_tb + num_c_t*(mean_c_t(:,c)-mean_t)*(mean_c_t(:,c)-mean_t)';
    S_tw = S_tw + Xt(:,index_t)*H_c_t*Xt(:,index_t)';
end

Sb = [S_sb,zeros(ds,dt);zeros(dt,ds),S_tb];
Sw = [S_sw,zeros(ds,dt);zeros(dt,ds),S_tw];

%% optimize the problem

A = X*(M+alpha*L)*X'+gamma*eye(size(X,1))+lamda*Sw;
B = beta*C+lamda*Sb;

[W,~] = eigs(A,B,dim,'sm');

Zx = W'*X;
Zx = Zx*diag(sparse(1./sqrt(sum(Zx.^2))));
Zs = Zx(:,1:ns);
Zt = Zx(:,(ns+1):end);

mmd_new = norm(mean(Zs,2)-mean(Zt,2));

%% Get the new data and train the classifier

model = svmtrain([Ys;Ytp],Zx(:,1:(ns+np))','-t 0 -q');
[pred_tar,acc_tar,~] = svmpredict(Yt((np+1):end),Zt(:,(np+1):end)', model, '-q');
acc_tar = acc_tar(1);

end


