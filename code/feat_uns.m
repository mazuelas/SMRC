function x_feat_uns = feat_uns(x,params)

n=length(x(1,:));
K2=length(params.C(:,1));

for i=1:n
    xx(:,i)=zeros(K2,1);
    for j=1:K2
        dist(j)=norm(x'-params.C(j,:));
    end
    [~,idx]=min(dist);
    xx(idx,i)=1;
end

xxx=params.coeff*(x-repmat(params.mu,1,n));

x_feat_uns=[ones(1,n);xxx;xx];

