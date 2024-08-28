
clear all 
close all

%Load data:

cd ..
cd data
load('usps.mat')
cd ..
cd code


n=length(y); 
d=length(x(:,1)); 

x=zscore(x')';


Y=length(unique(y));
I=eye(Y);

n_sup=500;
n_uns=7000;
n_test=1000;
K1=50;
K2=50;
K3=200;
K4=200;

%Get samples partitions:

c=cvpartition(y,'HoldOut',n_sup);
idx_sup=find(test(c)==1)';
idx_rest=find(training(c)==1)';

c2=cvpartition(y(idx_rest),'HoldOut',n_uns);
idx_uns=idx_rest(find(test(c2)==1));
idx_rest2=idx_rest(find(training(c2)==1));

c3=cvpartition(y(idx_rest2),'HoldOut',n_test);
idx_test=idx_rest2(find(test(c3)==1));

n_sup=length(idx_sup); 
n_uns=length(idx_uns);
n_test=length(idx_test);

%Use supervised and unsupervised learning algorithms:

param_uns=learn_uns(x(:,idx_uns),K1,K2);
param_sup=learn_sup(x(:,idx_sup),y(idx_sup),K3,K4);

%Obtain feature mapping and mean/confidence vectors:

phi_u=compute_phi_uns(x(:,[idx_sup,idx_uns]),param_uns,Y);
m_u=(K1+K2+1)*Y;

feat=zeros(m_u,n_sup);
for i=1:n_sup
    feat(:,i)=phi_u{i}(y(idx_sup(i)),:)';
end

tau_u=mean(feat,2);
lambda_u=std(feat,[],2)/sqrt(n_sup);

[phi_s,pred,score]=compute_phi_sup(x(:,[idx_sup,idx_uns]),param_sup,Y);
m_s=(K3)*Y;

feat=zeros(m_s,n_uns);
for i=1:n_uns
    feat(:,i)=phi_s{n_sup+i}(pred(n_sup+i),:)';
end

tau_s=mean(feat,2);
lambda_s=std(feat,[],2)/sqrt(n_uns);

tau=[tau_u;tau_s];
lambda=[lambda_u;lambda_s];
for i=1:n_sup+n_uns
    phi_semi{i}=horzcat(phi_u{i},phi_s{i});
end

% Increase confidence vector lambda to ensure the uncertainty set is not empty:

p0=zeros(Y*(n_sup+n_uns),1);

for i=1:n_sup
    p0((i-1)*Y+y(idx_sup(i)),1)=1;
end

for i=1:n_uns
    p0(n_sup*Y+(i-1)*Y+pred(n_sup+i),1)=1;
end

tau_rec=zeros(1,length(tau));

for i=1:n_sup+n_uns
    tau_rec=tau_rec+(p0((i-1)*Y+1:i*Y)'/(sum(p0)))*phi_semi{i};
end

lambda_aux=abs(tau-tau_rec');
for i=1:length(lambda)
     if lambda(i,1)<lambda_aux(i,1)
        lambda(i,1)=lambda_aux(i,1);
     end
end

%Obtain SMRC parameters and minimax risk:

iter=10^7;
B=1;
[muu,minimax]=fit_SMRC(tau,lambda,phi_semi,Y,iter,B);

%Estimate error using the test samples:

error=0;
error_sup=0;

for i=1:n_test

    aux=compute_phi_uns(x(:,idx_test(i)),param_uns,Y);
    aux2=compute_phi_sup(x(:,idx_test(i)),param_sup,Y);

    phi_x=horzcat(aux{1},aux2{1});
    [pred_SMRC,pred_det_SMRC,h]=predict_SMRC(phi_x,muu,Y);

    if pred_det_SMRC~=y(idx_test(i))
        error=error+1/n_test;
    end
    
    pred_sup=predict(param_sup.model,x(:,idx_test(i))');

    if pred_sup~=y(idx_test(i))
        error_sup=error_sup+1/n_test;
    end

end

