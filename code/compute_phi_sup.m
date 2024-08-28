function [phi,y_pred,prob] = compute_phi_sup(x,param,Y)

n=length(x(1,:));
I=eye(Y);

for i=1:n

[y_pred(i),prob(i,:)]=predict(param.model,x(:,i)');

xx=feat_sup(x(:,i),param);

    for k=1:Y
        if prob(i,y_pred(i))>param.th
            N(k,:)=kron(I(:,k),xx)';
        else
            N(k,:)=kron(ones(Y,1),xx)';
        end
    end
phi{i}=N;
N=[];

end
prob=prob';
end