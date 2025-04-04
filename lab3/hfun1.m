% Oblicza funkcję hipotezy regresji logistyczną (wielomianowej) przy użyciu
% macierzy wag W, stosując funkcję softmax
function evals = hfun1(W,features)
    % temp = exp(features * W) ; % ryzyko nadmiaru!
    % Lepiej.... (ale zamieniamy na ryzyko nieodmiaru)
    fw = features*W ;
    fw = fw - max(fw,[],2) ;
    temp = exp(fw) ;
    evals = temp ./ sum(temp,2) ;
end