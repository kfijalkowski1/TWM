%% Dobór parametrów klasyfikatorów
%% Parametry działania
% Powtarzalne wyniki
close all ;
rng('default') ;

% Liczba obrazów treningowych na klasę
cnt_train = 70 ;

% Liczba obrazów testowych na klasę
cnt_test = 30;

% Wybrane klasy obiektów
img_classes = {'deli', 'greenhouse', 'bathroom'};

% Liczba cech wybierana na każdym obrazie
feats_det = 100;

% Metoda wyboru cech (true - jednorodnie w całym obrazie, false - najsilniejsze)
feats_uniform = true;

% Wielkość słownika
words_cnt = 30 ;

% Detekcja cech
% Ładowanie pełnego zbioru danych z automatycznym podziałem na klasy
% Zbiór danych pochodzi z publikacji: A. Quattoni, and A.Torralba. <http://people.csail.mit.edu/torralba/publications/indoor.pdf 
% _Recognizing Indoor Scenes_>. IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2009.
% 
% Pełny zbiór dostępny jest na stronie autorów: <http://web.mit.edu/torralba/www/indoor.html 
% http://web.mit.edu/torralba/www/indoor.html>

imds_full = imageDatastore("C:\Users\Użytkownik WEiTI\Desktop\twm_lab3\indoor_images", "IncludeSubfolders", true, "LabelSource", "foldernames");
%countEachLabel(imds_full)

% Wybór przykładowych klas i podział na zbiór treningowy i testowy
[imds, imtest] = splitEachLabel(imds_full, cnt_train, cnt_test, 'Include', img_classes);
%countEachLabel(imds)

% Wyznaczenie punktów charakterystycznych we wszystkich obrazach zbioru treningowego
files_cnt = length(imds.Files);
all_points = cell(files_cnt, 1);
total_features = 0;

for i=1:files_cnt
    I = readImage(imds.Files{i});
    all_points{i} = getFeaturePoints(I, feats_det, feats_uniform);
    total_features = total_features + length(all_points{i});
end

% Przygotowanie listy przechowującej indeksy plików i punktów charakterystycznych
file_ids = zeros(total_features, 2);
curr_idx = 1;
for i=1:files_cnt
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 1) = i;
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 2) = 1:length(all_points{i});
    curr_idx = curr_idx + length(all_points{i});
end

% Obliczenie deskryptorów punktów charakterystycznych
all_features = zeros(total_features, 64, 'single');
curr_idx = 1;
for i=1:files_cnt
    I = readImage(imds.Files{i});
    curr_features = extractFeatures(rgb2gray(I), all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

% Tworzenie słownika

% Klasteryzacja punktów 
[idx, words, sumd, D] = kmeans(all_features, words_cnt, "MaxIter", 10000);
% Wizualizacja wyliczonych słów

% Wyznaczenie histogramów słów dla każdego obrazu treningowego
file_hist = zeros(files_cnt, words_cnt);
for i=1:files_cnt
    file_hist(i,:) = histcounts(idx(file_ids(:,1) == i), (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

% Wyznaczenie histogramów słów dla każdego obrazu testowego
test_hist = zeros(length(imtest.Files), words_cnt);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractFeatures(rgb2gray(I), pts);
    test_hist(i,:) = wordHist(feats, words);
end


%% Funkcja hipotezy i kosztu w klasyfikatorze regresji logistycznej
close all ;
rng('default') ;

% Przygotowujemy dane - uzupełniamy o dodatkową cechę (stałą 1) dla uproszczenia obliczeń 
featurestrain = file_hist ;
featurestrain = [featurestrain, ones(size(featurestrain,1),1)] ;

% Testujemy wyniki funkcji hipotezy h(x) = softmax(x * W) dla losowych wag (31
% cech x 3 klasy wyjściowe)
W = randn(31,3) 
featurestrain * W

% Po zastosowaniu funkcji softmax otrzymujemy estymowaną przynależność do
% klasy
res = hfun1(W,featurestrain)
[~,sellabel] = max(res,[],2)

% Funkcja oceny określa, jak bardzo wygenerowane wartośći przynależności
% pasują do danych uczących
labelstrain = imds.Labels == unique(imds.Labels)' % wektor etykiet w postaci one-hot
costfun1(W(:),featurestrain,labelstrain, 0.0)

%% Klasyfikator Regresji Logistycznej dla n-klas - pierwsza przymiarka
% Zadanie 1 -> Uruchom proces uczenia klasyfikatora. Podaj i zinterpretuj
% wyniki na zbiorze uczącym i testowym
close all
rng('default') ;

[W, cost] = trainsimple(featurestrain, imds.Labels, 0.0000, 'Iter') ;
labelstrain = imds.Labels == unique(imds.Labels)' ;

% Wyniki dla zbioru uczącego
pred = hfun1(W,featurestrain) ;
acc = getAccuracy(pred,labelstrain) * 100 ;
    
% Wyniki dla zbioru testowego
featurestest = test_hist ;
featurestest = [featurestest, ones(size(featurestest,1),1)] ;
labelstest = imtest.Labels == unique(imtest.Labels)' ;
predtest = hfun1(W,featurestest) ;
acctest = getAccuracy(predtest,labelstest) * 100 ;
 
fprintf(1,'Accuracy train: %f\n', acc) ;
fprintf(1,'Accuracy test: %f\n', acctest) ;

 
%% Klasyfikator Regresji Logistycznej dla n-klas - modyfikacja liczby cech (bez regularyzacji)
% Zadanie 2 - Uruchom proces uczenia klasyfikatora dla różnej liczby cech
% (i wag modelu). Opisz, w jaki sposób skuteczność i funkcja kosztu dla 
% zbioru uczącego i walidacyjnego zmienia się wraz ze zwiększaniem liczby cech 
% (i złożoności modelu). Zinterpretuj wynik. Zaproponuj dobór liczby cech w
% tym scenariuszu.
close all ;
rng('default') ;

feat_counts = 2:2:size(file_hist,2) ;

costsall = [] ;
costsstd = [] ;
costsvalall = [] ;
costsvalstd = [] ;
accsall = [] ;
accsstd = [] ;
accsvalall = [] ;
accsvalstd = [] ;
for cnt_usedfeatures = feat_counts    
    features = file_hist(:,1:cnt_usedfeatures) ;
    features = [features, ones(size(features,1),1)] ;
    
    [Ws, costs, costsval, accs, accsval] = crossval(features, imds.Labels, 0) ;
    costsall = [costsall, mean(costs)] ;
    costsstd = [costsstd, std(costs)] ;
    costsvalall = [costsvalall, mean(costsval(~isnan(costsval)))] ;
    costsvalstd = [costsvalstd, std(costsval(~isnan(costsval)))] ;
    accsall = [accsall, mean(accs)] ;
    accsstd = [accsstd, std(accs)] ;
    accsvalall = [accsvalall, mean(accsval)] ;
    accsvalstd = [accsvalstd, std(accsval)] ;
end

% % Rysowanie kosztów
% figure
% plot(feat_counts, costsall, feat_counts, costsvalall, 'LineWidth', 2) ;
% ylim([0 6])
% title('Funkcja kosztu vs. liczba cech') ;
% xlabel('Liczba cech') ; 
% ylabel('Funkcja kosztu') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie niepewności estymacji kosztów
% figure
% plot(feat_counts, costsstd, feat_counts, costsvalstd, 'LineWidth', 2) ;
% title('Odch. std. funkcji kosztu vs. liczba cech') ;
% xlabel('Liczba cech') ;
% ylabel('Odch. std. funkcji kosztu') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie skuteczności
% figure
% plot(feat_counts, accsall, feat_counts, accsvalall, 'LineWidth', 2) ;
% title('Skuteczność vs. liczba cech') ;
% xlabel('Liczba cech') ;
% ylabel('Skuteczność (%)') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie niepewności estymacji skuteczności
% figure
% plot(feat_counts, accsstd, feat_counts, accsvalstd, 'LineWidth', 2) ;
% title('Odch. std. skuteczności vs. liczba cech') ;
% xlabel('Liczba cech') ;
% ylabel('Odch. std. skuteczności') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;

%% Klasyfikator Regresji Logistycznej dla n-klas - modyfikacja wielkości zbioru danych (bez regularyzacji)
% Zadanie 3 - Uruchom proces uczenia klasyfikatora dla rosnącej wielkości
% zbioru uczącego (przy zachowaniu tego samego modelu). Zinterpretuj
% zmiany skuteczności i funkcji kosztu na zbiorze uczącym i
% walidacyjnym

close all ;
rng('default') ;

%feat_counts = 1:2:size(file_hist,2) ;

costsall = [] ;
costsstd = [] ;
costsvalall = [] ;
costsvalstd = [] ;
accsall = [] ;
accsstd = [] ;
accsvalall = [] ;
accsvalstd = [] ;

subset_sizes = [] ;
num_features = 5 ;
for subset_prop = 0.2:0.1:0.9 
    part = cvpartition(imds.Labels, 'HoldOut', 1-subset_prop) ;
    
    features = file_hist(part.training(1),1:num_features) ;    
    labels = imds.Labels(part.training(1)) ;
    features = [features, ones(size(features,1),1)] ;
    subset_sizes = [subset_sizes, part.TrainSize] ;
    
    [Ws, costs, costsval, accs, accsval] = crossval(features, labels, 0, 10) ;
    
    costsall = [costsall, mean(costs)] ;
    costsstd = [costsstd, std(costs)] ;
    costsvalall = [costsvalall, mean(costsval(~isnan(costsval)))] ;
    costsvalstd = [costsvalstd, std(costsval(~isnan(costsval)))] ;
    accsall = [accsall, mean(accs)] ;
    accsstd = [accsstd, std(accs)] ;
    accsvalall = [accsvalall, mean(accsval)] ;
    accsvalstd = [accsvalstd, std(accsval)] ;
end

% Rysowanie kosztów
% figure
% plot(subset_sizes, costsall, subset_sizes, costsvalall, 'LineWidth', 2) ;
% title(strcat('Funkcja kosztu vs. wielkość zbioru danych numfeatures = ',num2str(num_features))) ;
% xlabel('Wielkość zbioru') ; 
% ylabel('Funkcja kosztu') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie niepewności estymacji kosztów
% figure
% plot(subset_sizes, costsstd, subset_sizes, costsvalstd, 'LineWidth', 2) ;
% title(strcat('Odch. std. funkcji kosztu vs. wielkość zbioru danych numfeatures = ',num2str(num_features))) ;
% xlabel('Wielkość zbioru') ; 
% ylabel('Odch. std. funkcji kosztu') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie skuteczności
% figure
% plot(subset_sizes, accsall, subset_sizes, accsvalall, 'LineWidth', 2) ;
% title(strcat('Skuteczność vs. wielkość zbioru danych numfeatures = ',num2str(num_features))) ;
% xlabel('Wielkość zbioru') ;
% ylabel('Skuteczność (%)') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie niepewności estymacji skuteczności
% figure
% plot(subset_sizes, accsstd, subset_sizes, accsvalstd, 'LineWidth', 2) ;
% title(strcat('Odch. std. skuteczności vs. wielkość zbioru danych numfeatures = ',num2str(num_features))) ;
% xlabel('Wielkość zbioru') ;
% ylabel('Odch. std. skuteczności') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;

%% Klasyfikator Regresji Logistycznej dla n-klas - wczesne zatrzymanie (bez regularyzacji)
% Zadanie 4 - Uruchom proces uczenia klasyfikatora w scenariuszu wcześniejszego 
% zatrzymania procesu uczenia. Zinterpretuj zachowanie funkcji kosztu i
% skuteczności dla różnej liczby iteracji. Wybierz iterację, po której
% najlepiej zatrzymać proces uczenia (i zachowaj w zmiennej sel_iter) -
% wybór uzasadnij. Jak tak nauczony klasyfikator zachowuje się na zbiorze
% testowym?
close all ;
rng('default') ;

cnt_usedfeatures = size(file_hist,2) ;

features = file_hist(:,1:cnt_usedfeatures) ;
features = [features, ones(size(features,1),1)] ;
[Ws, costs, costsval, accs, accsval] = crossvalearly(features, imds.Labels, 0) ;
    
% Rysowanie kosztów
% figure
% %plot(1:length(costs), costs, 1:length(costs), costsval, 'LineWidth', 2) ;
% plot(1:length(costs), costs, 1:length(costs), costsval, 'LineWidth', 2) ;
% title('Funkcja kosztu vs. liczba iteracji optymalizacji funkcji kosztu') ;
% xlabel('Liczba iteracji') ; 
% ylabel('Funkcja kosztu') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% ylim([0,4]) ; %Lepsza widoczność
% 
% % Rysowanie skuteczności
% figure
% plot(1:length(accs), accs, 1:length(accsval), accsval, 'LineWidth', 2) ;
% title('Skuteczność vs. liczba iteracji optymalizacji funkcji kosztu') ;
% xlabel('Liczba iteracji') ; 
% ylabel('Skuteczność') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Wyniki dla zbioru uczącego
sel_iter = 25 % Do wypełnienia w ramach zadania 
pred = hfun1(Ws(:,:,sel_iter),featurestrain) ;
acc = getAccuracy(pred,labelstrain) * 100 ;

% Wyniki dla zbioru testowego
featurestest = test_hist ;
featurestest = [featurestest, ones(size(featurestest,1),1)] ;
labelstest = imtest.Labels == unique(imtest.Labels)' ;
predtest = hfun1(Ws(:,:,sel_iter),featurestest) ; % Wybór wag dla iteracji sel_iter
acctest = getAccuracy(predtest,labelstest) * 100 ;

fprintf(1,'Accuracy train: %f\n', acc) ;
fprintf(1,'Accuracy test: %f\n', acctest) ;

%% Klasyfikator Regresji Logistycznej dla n-klas - regularyzacja z normą L2
% Zadanie 5 - Uruchom proces uczenia klasyfikatora kilkukrotnie i dobierz
% dobrą wartość współczynnia regularyzacji lambda, na podstawie analizy
% skuteczności oraz funcji kosztu dla różnych wartości tego parametru.
% Zagęść kilkukrotnie siatkę podziału w celu lepszej lokalizacji parmetru
% lambda. Podaj wybraną wartość parametru i uzasadnij wybór.

close all ;
rng('default') ;

%feat_counts = 1:2:size(file_hist,2) ;

costsall = [] ;
costsstd = [] ;
costsvalall = [] ;
costsvalstd = [] ;
accsall = [] ;
accsstd = [] ;
accsvalall = [] ;
accsvalstd = [] ;

lambdas = logspace(-4,-1,15) ;
%lambdas = logspace(...) ; % Zagęszczenie próby...
%lambdas = logspace(...) ; % Zagęszczenie próby...
%lambdas = logspace(...) ; % Zagęszczenie próby...
features = file_hist ;
features = [features, ones(size(features,1),1)] ;
for lambda = lambdas   
    
    subset_sizes = [subset_sizes, part.TrainSize] ;
    
    [Ws, costs, costsval, accs, accsval] = crossval(features, imds.Labels, lambda, 10) ;
    
    costsall = [costsall, mean(costs)] ;
    costsstd = [costsstd, std(costs)] ;
    costsvalall = [costsvalall, mean(costsval(~isnan(costsval)))] ;
    costsvalstd = [costsvalstd, std(costsval(~isnan(costsval)))] ;
    accsall = [accsall, mean(accs)] ;
    accsstd = [accsstd, std(accs)] ;
    accsvalall = [accsvalall, mean(accsval)] ;
    accsvalstd = [accsvalstd, std(accsval)] ;
end

% % Rysowanie kosztów
% figure
% semilogx(lambdas, costsall, lambdas, costsvalall, 'LineWidth', 2) ;
% title('Funkcja kosztu vs. lambda') ;
% xlabel('Lambda') ; 
% ylabel('Funkcja kosztu') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie niepewności estymacji kosztów
% figure
% semilogx(lambdas, costsstd, lambdas, costsvalstd, 'LineWidth', 2) ;
% title('Odch. std. funkcji kosztu vs. lambda') ;
% xlabel('Lambda') ; 
% ylabel('Odch. std. funkcji kosztu') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie skuteczności
% figure
% semilogx(lambdas, accsall, lambdas, accsvalall, 'LineWidth', 2) ;
% title('Skuteczność vs. lambda') ;
% xlabel('Lambda') ;
% ylabel('Skuteczność (%)') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;
% 
% % Rysowanie niepewności estymacji skuteczności
% figure
% semilogx(lambdas, accsstd, lambdas, accsvalstd, 'LineWidth', 2) ;
% title('Odch. std. skuteczności vs. lambda') ;
% xlabel('Lambda') ;
% ylabel('Odch. std. skuteczności') ;
% legend('Zb. treningowy','Zb. walidacyjny') ;

%% Klasyfikator Regresji Logistycznej dla n-klas - z regularyzacją
% Zadanie 6 - dla wybranej wartości parametru lambda przeprowadź uczenie na
% pełnym zbiorze uczącym i zaraportuj wyniki na zbiorze testowym
close all ;
rng('default') ;

lambda = 0.002; % Do wypełnienia w ramach zadania
featurestrain = file_hist ;
featurestrain = [featurestrain, ones(size(featurestrain,1),1)] ;
[W, cost] = trainsimple(featurestrain, imds.Labels, lambda) ;
labelstrain = imds.Labels == unique(imds.Labels)' ;

% Wyniki dla zbioru uczącego
pred = hfun1(W,featurestrain) ;
acc = getAccuracy(pred,labelstrain) * 100 ;
    
% Wyniki dla zbioru testowego
featurestest = test_hist ;
featurestest = [featurestest, ones(size(featurestest,1),1)] ;
labelstest = imtest.Labels == unique(imtest.Labels)' ;
predtest = hfun1(W,featurestest) ;
acctest = getAccuracy(predtest,labelstest) * 100 ;
 
fprintf(1,'Accuracy train: %f\n', acc) ;
fprintf(1,'Accuracy test: %f\n', acctest) ;


%% Funkcje pomocnicze

% Optymalizacja bez ograniczeń z restartami
function [Wout, costout] = optimize1(features,labels)
    Wout = [] ;
    costout = inf ;
    for i=1:10
        W = rand(size(features,2), size(labels,2)) ;
        %cost = costfun1(W(:),features,labels) ;
        options = optimoptions('fminunc','Display','iter', 'MaxFunctionEvaluations', 10e9) ;
        [w,cost,exitflag] = fminunc(@(w) costfun1(w,features,labels),W(:),options) ;        
        if exitflag > 0
             Wout = reshape(w,size(features,2), size(labels,2)) ;
             costout = cost ;
        end
    end
end

function P = getPatch(I, pt, scale, scale_factor)
    x1 = round(pt(1) - 0.5*scale*scale_factor);
    x2 = round(pt(1) + 0.5*scale*scale_factor);
    y1 = round(pt(2) - 0.5*scale*scale_factor);
    y2 = round(pt(2) + 0.5*scale*scale_factor);
    
    [x1, x2, y1, y2] = clipInside(x1, x2, y1, y2, size(I, 1), size(I, 2));
    
    P = imresize(I(y1:y2, x1:x2, :), [64 64]);
end

function [xr1, xr2, yr1, yr2] = clipInside(x1, x2, y1, y2, rows, cols)
    xr1 = min(max(x1, 1), cols);
    xr2 = min(max(x2, 1), cols);
    yr1 = min(max(y1, 1), rows);
    yr2 = min(max(y2, 1), rows);
end

function pts = getFeaturePoints(I, pts_det, pts_uniform)
    if size(I, 3) > 1
        I2 = rgb2gray(I);
    else
        I2 = I;
    end
    
    pts = detectSURFFeatures(I2, 'MetricThreshold', 100);
    if pts_uniform
        pts = selectUniform(pts, pts_det, size(I));
    else
        pts = pts.selectStrongest(pts_det);
    end
end

function h = wordHist(feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [~, lbl] = min(dis, [], 2);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

function [h, P] = visSingleImage(I, pts, feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [dis, lbl] = min(dis, [], 2);
    [~, ids] = sort(dis);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
    P = zeros(words_cnt*64, 30*64, 3, 'uint8');
    pos = zeros(words_cnt, 1);
    for i=1:size(feats, 1)
        id = ids(i);
        x = pos(lbl(id)) * 64;
        pos(lbl(id)) = min(pos(lbl(id)) + 1, 29);
        y = (lbl(id)-1) * 64;
        pat = getPatch(I, pts.Location(id, :), pts.Scale(id), 12);
        pat = insertText(pat, [2, 2], dis(id), 'FontSize', 10, 'BoxOpacity', 0);
        pat = insertText(pat, [1, 1], dis(id), 'FontSize', 10, 'BoxOpacity', 0, 'TextColor', 'white');
        P(y+1:y+64, x+1:x+64, :) = pat;
    end
end

% Wczytanie obrazu i przeskalowanie jeśli jest zbyt duży
function I = readImage(path)
    I = imread(path);
    if size(I,2) > 640
        I = imresize(I, [NaN 640]);
    end
end
