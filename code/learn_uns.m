function param=learn_uns(x_uns,K1,K2)

[coeff,score,~,~,~,mu]=pca(x_uns');

for i=1:K1
    param.coeff(i,:)=(coeff(:,i)')/std(score(:,i));
end

param.mu=mu';

[~,param.C]=kmeans(x_uns',K2);



