function [Xs,Ys,Xt,Yt,Xsp,Ysp,Xtp,Ytp] = load_data(name,paired_percent)

load(name);

Xs = data_src;
Xt = data_tar;
Ys = labels;
Yt = labels;

n_examples = size(data_src,1);
n_paired = floor(n_examples*paired_percent);
Xsp = data_src(1:n_paired,:);
Xtp = data_tar(1:n_paired,:);
Ysp = labels(1:n_paired,:);
Ytp = labels(1:n_paired,:);

end

