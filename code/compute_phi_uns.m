function phi = compute_phi_uns(x,param,Y)

n=length(x(1,:));
I=eye(Y);

for i=1:n
xx=feat_uns(x(:,i),param);
    for k=1:Y
    N(k,:)=kron(I(:,k),xx)';
    end
phi{i}=N;
end
 




