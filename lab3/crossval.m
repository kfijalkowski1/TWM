function [Ws, costs, costsval, accs, accsval] = crossval(inputfeatures, inputlabels, regcoeff, kfold)

if nargin < 3
    regcoeff = 0 ;
end

if nargin < 4
    kfold = 10 ;
end

part = cvpartition(inputlabels, 'kfold', kfold) ;
labels = inputlabels == unique(inputlabels)' ;

costs = [] ;
costsval = [] ;
accs = [] ;
accsval = [] ;
Ws = [] ;
fprintf(1, 'Obliczenia dla %d cech\n',size(inputfeatures,2)) ;
for i = 1:part.NumTestSets
    featurestrain = inputfeatures(part.training(i),:);
    labelstrain = labels(part.training(i),:) ;
    
    featuresval = inputfeatures(part.test(i),:) ;
    labelsval = labels(part.test(i),:) ;
    
    % Minimalizacja funkcji
    W = rand(size(featurestrain,2), size(labelstrain,2)) ;
    options = optimoptions('fminunc','Display','none', 'MaxFunctionEvaluations', 10e9) ;
    [w,cost,exitflag] = fminunc(@(w) costfun1(w,featurestrain,labelstrain,regcoeff),W(:),options) ;   
    W = reshape(w,size(featurestrain,2), size(labelstrain,2)) ;
    
    % Wyniki dla zbioru uczącego
    cost = costfun1(W(:),featurestrain,labelstrain, 0) ; % Dla raportowania użyjmy 'czystej' funkcji kosztu - regcoeff = 0
    pred = hfun1(W,featurestrain) ;
    acc = getAccuracy(pred,labelstrain) * 100 ;
    
    % Wyniki dla zbioru walidacyjnego
    costval = costfun1(W(:),featuresval,labelsval, 0) ; % Dla raportowania użyjmy 'czystej' funkcji kosztu - regcoeff = 0
    predval = hfun1(W,featuresval) ;
    accval = getAccuracy(predval,labelsval) * 100 ;
    
    Ws = cat(3,Ws,W) ;
    % Zapamiętujemy najlepszy
%     if cost < mincostval
%         mincostval = costval ;
%         mincost = cost ;
%         Ws = W ;
%     end
    
    % Statystyki
    costs = [costs,cost] ;
    costsval = [costsval,costval] ;
    accs = [accs,acc] ;
    accsval = [accsval,accval] ;
end
%coststotal = median(costs) ;
%costsvaltotal = median(costsval(~isnan(costsval))) ;
%costsval
%assert(std(coststestl(~isnan(coststestl))) < 1) ;
%accstotal = mean(accs) ;
%accsvaltotal = mean(accsval) ;
%accsvalstd = [accsvalstd,std(accsval)] ;