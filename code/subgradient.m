function [grad,varphi,chi]= subgradient(phi_x,muu,Y)

v=phi_x*muu;

[~,ord]=sort(v,'descend');

varphi=v(ord(1))-1;
chi=0;
grad=phi_x(ord(1),:)';

for size=2:Y
value_new=((size-1)*varphi+v(ord(size)))/(size);

if value_new>=varphi
    varphi=value_new;
    chi=varphi-(v(ord(1))-1);
    grad=((size-1)*grad+phi_x(ord(size),:)')/size;
else
    break
end
end


