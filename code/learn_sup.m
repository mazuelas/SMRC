function param=learn_sup(x_sup,y_sup,K3,K4)


param.model=fitcensemble(x_sup',y_sup,'ScoreTransform','doublelogit','NumLearningCycles',K4);
model_aux=fitcensemble(x_sup',y_sup,'ScoreTransform','doublelogit','CrossVal','on','KFold',10,'NumLearningCycles',K4);

err=kfoldLoss(model_aux);
[pred,scores]=kfoldPredict(model_aux);
for i=1:length(pred)
score(i)=scores(i,pred(i));
end

param.th=prctile(score,err*100)-10^-4;

mod=fitcnet(x_sup',y_sup,'LayerSizes',K3);

W=mod.LayerWeights{1};
F=W*x_sup;
sigma=std(F,[],2);
param.W=W;
param.mu=mean(F,2);
param.D=diag(1./sigma);




