function [Ws, costs, costsval, accs, accsval] = crossvalearly(inputfeatures, inputlabels, regcoeff)

ws = [] ;

part = cvpartition(inputlabels, 'HoldOut', 0.3) ;
labels = inputlabels == unique(inputlabels)' ;

costs = [] ;
costsval = [] ;
accs = [] ;
accsval = [] ;
Ws = [] ;
%fprintf(1, 'Obliczenia dla %d cech\n',size(inputfeatures,2)) ;

featurestrain = inputfeatures(part.training(1),:);
labelstrain = labels(part.training(1),:) ;

featuresval = inputfeatures(part.test(1),:) ;
labelsval = labels(part.test(1),:) ;

% Minimalizacja funkcji
W = rand(size(featurestrain,2), size(labelstrain,2)) ;
options = optimoptions('fminunc','Display','none', 'MaxFunctionEvaluations', 10e9, 'OutputFcn', @outfun) ;

[w,cost,exitflag] = fminunc(@(w) costfun1(w,featurestrain,labelstrain,regcoeff),W(:),options) ;
%W = reshape(w,size(featurestrain,2), size(labelstrain,2)) ;

% Śledzenie historii obliczeń
for i=1:size(ws,2)
    w = ws(:,i) ;
    W = reshape(w,size(featurestrain,2), size(labelstrain,2)) ;
    cost = costfun1(W(:),featurestrain,labelstrain,0) ; % Dla raportowania użyjmy czystej funkcji kosztu (regcoeff = 0)
    pred = hfun1(W,featurestrain) ;
    acc = getAccuracy(pred,labelstrain) * 100 ;
    costval = costfun1(W(:),featuresval,labelsval,0) ; % Dla raportowania użyjmy funkcji czystej funkcji kosztu (regcoeff = 0)
    predval = hfun1(W,featuresval) ;
    accval = getAccuracy(predval,labelsval) * 100 ;
    Ws = cat(3,Ws,W) ;
    
    % Statystyki
    costs = [costs,cost] ;
    costsval = [costsval,costval] ;
    accs = [accs,acc] ;
    accsval = [accsval,accval] ;
end




% Zapamiętujemy najlepszy
%     if cost < mincostval
%         mincostval = costval ;
%         mincost = cost ;
%         Ws = W ;
%     end



%coststotal = median(costs) ;
%costsvaltotal = median(costsval(~isnan(costsval))) ;
%costsval
%assert(std(coststestl(~isnan(coststestl))) < 1) ;
%accstotal = mean(accs) ;
%accsvaltotal = mean(accsval) ;
%accsvalstd = [accsvalstd,std(accsval)] ;

function stop = outfun(w,optimValues,state)
    ws = [ws,w] ;
    stop = false;
end


end



