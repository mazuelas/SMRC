function x_feat_sup = feat_sup(x,param)

n=length(x(1,:));

x_feat_sup=param.D*(param.W*x-repmat(param.mu,1,n));

