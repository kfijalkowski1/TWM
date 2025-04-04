function acc = getAccuracy(pred,labels)
    pred = (max(pred,[],2) == pred) ;
    hits = max(pred == labels & labels,[],2) ;
    acc = sum(hits)/length(hits) ;
end