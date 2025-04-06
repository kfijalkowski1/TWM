

% Oblicza funkcję kosztu przy użyciu funkcji entropii krzyżowej
function cost = costfun1(w,features,labels,lambda)
    W = reshape(w,size(features,2),size(labels,2)) ;
    uselogsumexptrick = true ;
    if ~uselogsumexptrick 
        evals = hfun1(W, features) ;
        logevals = log(evals) ;
    else
        % Używamy Log-Sum-Exp trick w celu uniknięcia nadmiaru/niedomiaru
        % przy potęgowaniu
        % https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        % https://feedly.com/engineering/posts/tricks-of-the-trade-logsumexp
        fw = features*W ; % funkcja hipotezy hfun1, ale bez aktywacji softmax
        offset = max(fw,[],2) ; % przesunięcie w celu uniknięcia nadmiaru przy exp(...)
        sumlog = log(sum(exp(fw-offset),2)) ;
        logevals = fw - sumlog - offset ;  % formuła Log-Sum-Exp-trick      
    end
    
    logloss = -labels .* logevals ;
    logloss(labels == 0) = 0 ; % unikamy NaN dla lograrytmów z 0
    cost = sum(sum(logloss))/size(features,1) ;
    cost = cost + lambda * sum(w(1:end-1).^2)/size(features,1) ;
    % cost = cost + lambda * sum(abs(w(1:end-1)))/size(features,1) ;
    %ws = W(1:end-1,:) ;
    %ws = ws(:) ;
    %cost = cost + 0.000001*(var(ws) - 1)^2;
end