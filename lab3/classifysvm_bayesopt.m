%% Operating parameters
% Reproducible results
close all ;
rng('default') ;

% Number of training images per class
cnt_train = 70 ;

% Number of test images per class
cnt_test = 30;

% Selected object classes
img_classes = {'deli', 'greenhouse', 'bathroom'};

% Number of features selected for each image
feats_det = 100;

% Feature selection method (true - uniformly across the entire image, false - strongest)
feats_uniform = true;

% Dictionary size
words_cnt = 30 ;

% Feature detection
% Loading the full dataset with automatic class division
% The dataset comes from the publication: A. Quattoni, and A.Torralba. <http://people.csail.mit.edu/torralba/publications/indoor.pdf 
% _Recognizing Indoor Scenes_>. IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2009.
% 
% The full dataset is available on the authors' website: <http://web.mit.edu/torralba/www/indoor.html 
% http://web.mit.edu/torralba/www/indoor.html>

imds_full = imageDatastore("indoor_images/", "IncludeSubfolders", true, "LabelSource", "foldernames");
%countEachLabel(imds_full)

% Selection of example classes and division into training and test sets
[imds, imtest] = splitEachLabel(imds_full, cnt_train, cnt_test, 'Include', img_classes);
%countEachLabel(imds)

% Determining feature points in all images of the training set
files_cnt = length(imds.Files);
all_points = cell(files_cnt, 1);
total_features = 0;

for i=1:files_cnt
    I = readImage(imds.Files{i});
    all_points{i} = getFeaturePoints(I, feats_det, feats_uniform);
    total_features = total_features + length(all_points{i});
end

% Preparing a list storing file indices and feature points
file_ids = zeros(total_features, 2);
curr_idx = 1;
for i=1:files_cnt
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 1) = i;
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 2) = 1:length(all_points{i});
    curr_idx = curr_idx + length(all_points{i});
end

% Calculating feature point descriptors
all_features = zeros(total_features, 64, 'single');
curr_idx = 1;
for i=1:files_cnt
    I = readImage(imds.Files{i});
    curr_features = extractFeatures(rgb2gray(I), all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

% Creating a dictionary

% Clustering points
[idx, words, sumd, D] = kmeans(all_features, words_cnt, "MaxIter", 10000);
% Visualization of calculated words

% Determining word histograms for each training image
file_hist = zeros(files_cnt, words_cnt);
for i=1:files_cnt
    file_hist(i,:) = histcounts(idx(file_ids(:,1) == i), (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

% Determining word histograms for each test image
test_hist = zeros(length(imtest.Files), words_cnt);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractFeatures(rgb2gray(I), pts);
    test_hist(i,:) = wordHist(feats, words);
end

%% SVM with Grid Search
fprintf('Starting Bayesian optimization for SVM parameters...\n');

% Define the objective function for bayesopt
ObjFcn = @(x)SVMCVObjective(x, file_hist, imds.Labels);

% Define optimization variables (parameters to optimize)
vars = [
    optimizableVariable('C', [1e-3, 1e3], 'Transform', 'log')
    optimizableVariable('gamma', [1e-3, 1e3], 'Transform', 'log')
];

% Run Bayesian optimization
results = bayesopt(ObjFcn, vars, ...
    'MaxObjectiveEvaluations', 30, ...
    'IsObjectiveDeterministic', false, ...
    'UseParallel', true, ...
    'Verbose', 1);

% Extract best parameters
best_params = results.XAtMinObjective;
best_C = best_params.C;
best_gamma = best_params.gamma;
best_accuracy = 1 - results.MinObjective;

% Helper function for SVM cross-validation
function error = SVMCVObjective(params, X, Y)
    % Create SVM template with current parameters
    temp = templateSVM('KernelFunction', 'gaussian', 'BoxConstraint', params.C, 'KernelScale', params.gamma);
    
    % Train the model
    model = fitcecoc(X, Y, 'Learners', temp);
    
    % Perform 5-fold cross-validation
    modelcv = crossval(model, 'KFold', 5);
    error = kfoldLoss(modelcv);
end
fprintf('\nBest parameters found: C=%f, gamma=%f, CV accuracy=%f\n', best_C, best_gamma, best_accuracy);

% Visualize grid search results
figure;
imagesc(log10(gamma_values), log10(C_values), results);
colorbar;
xlabel('log10(gamma)');
ylabel('log10(C)');
title('Grid Search Results: CV Accuracy');
set(gca, 'XTick', log10(gamma_values), 'XTickLabel', gamma_values);
set(gca, 'YTick', log10(C_values), 'YTickLabel', C_values);

% Train final model with best parameters
fprintf('Training final model with best parameters...\n');
temp = templateSVM('KernelFunction', 'gaussian', 'BoxConstraint', best_C, 'KernelScale', best_gamma);
model = fitcecoc(file_hist, imds.Labels, 'Learners', temp);

% Evaluate on training and test sets
train_err = loss(model, file_hist, imds.Labels, 'Lossfun', 'classiferror');
test_err = loss(model, test_hist, imtest.Labels, 'Lossfun', 'classiferror');
fprintf('Final model - train_acc: %f, test_acc: %f\n', 1-train_err, 1-test_err);

% Cross-validation of the final model
modelcv = crossval(model, 'KFold', 5);
cv_err = kfoldLoss(modelcv);
fprintf('Final model - cross_validation_accuracy: %f\n', 1-cv_err);


%% Helper functions

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

% Loading an image and resizing if it's too large
function I = readImage(path)
    I = imread(path);
    if size(I,2) > 640
        I = imresize(I, [NaN 640]);
    end
end