function [muu_sol,minimax] = fit_SMRC(tau,lambda,phi_x,Y,iter,B)

m=length(tau);
n=length(phi_x);

muu=sparse(zeros(m,1));
muu_aux=zeros(m,1);
muu_sol=zeros(m,1);
th=10^-6;


cont=1;



step=0.1/m;

while cont<iter

for i=1:B:n-B+1

  grad=zeros(m,1);
  for j=0:B-1
      grad=grad+(1/B)*subgradient(phi_x{i+j},muu,Y);
  end
  muu_aux=muu_aux-step*(-tau+lambda.*sign(muu_aux)+grad);

idx=find(abs(muu_aux)>th);
muu=sparse(idx,ones(1,length(idx)),muu_aux(idx),m,1);

  cont=cont+1;
    muu_sol=((cont-1)/cont)*muu_sol+(1/cont)*muu_aux;


end
end



value=0;

for i=1:n
    [~,varphi,~]=subgradient(phi_x{i},muu_sol,Y);
    value=value+(1/n)*varphi;
end

minimax=1-tau'*muu_sol+lambda'*abs(muu_sol)+value;



