function [y,y_det,h]=predict_SMRC(phi_x,muu,Y)

v=phi_x*muu;
vv=sort(v,'descend');


value=vv(1)-1;

size=1;
stop=0;

while stop==0 && size < Y

    value_new=(size*value+vv(size+1))/(size+1);

    if value_new<value
        stop=1;
    else
        size=size+1;
        value=value_new;
      end
end

h=pos(v-value);

y=find(mnrnd(1,h)==1);
[~,y_det]=max(h);
