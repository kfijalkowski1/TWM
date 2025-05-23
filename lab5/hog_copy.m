figure %[output:8c972d18]
img = rgb2gray(imread("grad.png"))
imshow(img, 'InitialMagnification', 1000); %[output:8c972d18]
%%
Gx = imfilter(double(img), [-1, 0, 1], 'replicate')
Gy = imfilter(double(img), [-1, 0, 1]', 'replicate')
Gp = sqrt(Gx.*Gx + Gy.*Gy)
Ga = rad2deg(atan2(Gy, Gx))
Ga(Ga < 0) = Ga(Ga < 0) + 180;
Ga(Ga == 180) = 0
Gd = round(Ga / 20);
Gd(Gd == 9) = 0
imagesc(Gp) %[output:6f936e34]
imagesc(Ga) %[output:65b7c763]
%%
ids = 0:8;
sums = zeros(1, 9);
for i=ids
    sums(i+1) = sum(Gp(Gd == i));
end

bar(ids, sums) %[output:7e856adc]
%%
figure; %[output:1eb600a4]
I1 = imread("person_1.jpg");
[hog1,vis1] = extractHOGFeatures(I1);
imshow(I1, 'InitialMagnification', 500);  %[output:1eb600a4]
hold on; %[output:1eb600a4]
plot(vis1); %[output:1eb600a4]
figure; %[output:94e419bc]
I2 = imread("person_2.jpg");
[hog2,vis2] = extractHOGFeatures(I2);
imshow(I2, 'InitialMagnification', 500);  %[output:94e419bc]
hold on; %[output:94e419bc]
plot(vis2); %[output:94e419bc]
%%
figure; %[output:8d5d7ab0]
It = imread("people_1.jpg");
imshow(It); %[output:8d5d7ab0]
hold on %[output:8d5d7ab0]
[hogt,vist] = extractHOGFeatures(It);
plot(vist) %[output:8d5d7ab0]
%%
sx = 48;
sy = 128;
step = 8;

scale_step = 0.8;
levels = 5;

scale = 1.0;
thr = 60;

dets = [];
scores = [];
labels = {};

for k=1:levels %[output:group:55d41b75]

scale %[output:8b1094de] %[output:8f90b176] %[output:91730be1] %[output:3228a6b8] %[output:7d1340d7]
cur_img = imresize(It, scale);

w = size(cur_img, 2);
h = size(cur_img, 1);

count_x = floor((w - sx) / step) %[output:2380968e] %[output:1afca9a6] %[output:2742c67d] %[output:45fc9547] %[output:0470e778]
count_y = floor((h - sy) / step) %[output:583e2d5b] %[output:90f8655d] %[output:7f38448c] %[output:593c9d24] %[output:5041e88e]

out = zeros(count_y, count_x);

for j=0:count_y-1
    for i=0:count_x-1
        x = 1+(i*step);
        y = 1+(j*step);
        sub_img = cur_img(y:y+sy, x:x+sx);
        sub_hog = extractHOGFeatures(sub_img);
        % klasyfikator
        dist = min(sum((sub_hog - hog1) .^ 2 ), sum((sub_hog - hog2) .^ 2 ));
        out(j+1, i+1) = dist;
    end
end

best_score = min(min(out)) %[output:85c26b49] %[output:04d5ccf6] %[output:688f9032] %[output:34a59360] %[output:517c4c3b]

figure; %[output:4c00cd6d] %[output:6af011ea] %[output:0bcf9c38] %[output:1f1079a1] %[output:163e0256]
imagesc(out) %[output:4c00cd6d] %[output:6af011ea] %[output:0bcf9c38] %[output:1f1079a1] %[output:163e0256]
figure; %[output:5e41d93c] %[output:525e289d] %[output:4813026b] %[output:13e1894b] %[output:8f1ad1dd]
mins = imregionalmin(out);
mins(out > thr) = 0;
imagesc(mins); %[output:5e41d93c] %[output:525e289d] %[output:4813026b] %[output:13e1894b] %[output:8f1ad1dd]
[rs, cs] = find(mins);

for i=1:size(rs)
    x = ((cs(i)-1) * step) / scale + 1;
    y = ((rs(i)-1) * step) / scale + 1;
    w = sx / scale;
    h = sy / scale;
    
    dets = [dets; [x,y,w,h] ];
    scores = [scores, out(rs(i), cs(i))];
    labels{size(dets, 1)} = num2str(out(rs(i), cs(i)), '%.1f');
end

scale = scale * scale_step;

end %[output:group:55d41b75]
%%
imshow(insertObjectAnnotation(It, 'rectangle', dets, labels)); %[output:281a1d85]
%%
%[text] ### Filtracja detekcji i NMS
filtered_dets = [];
filtered_scores = [];
tmp_dets = dets;
tmp_scores = scores;
while 1
    [m, i] = min(tmp_scores);
    filtered_dets = [filtered_dets; tmp_dets(i,:)];
    filtered_scores = [filtered_scores, m];
    ratio = bboxOverlapRatio(tmp_dets(i, :), tmp_dets);
    tmp_dets = tmp_dets(ratio < 0.2, :);
    tmp_scores = tmp_scores(ratio < 0.2);
    if size(tmp_dets, 1) < 1
        break
    end
end
cnt = size(filtered_scores, 2);
filtered_lbls = cell(cnt, 1);
for i=1:cnt
    filtered_lbls{i} = num2str(filtered_scores(i), '%.1f');
end
imshow(insertObjectAnnotation(It, "rectangle", filtered_dets, filtered_lbls)); %[output:8b8d04ea]
%%
gt_rect=gTruth.LabelData.person{1,1};
ann1 = insertObjectAnnotation(It, "rectangle", gt_rect, "person");
imshow(ann1); %[output:1518dfba]
score_thr = 30;
selected_dets = filtered_dets(filtered_scores < score_thr, :);
ratio = bboxOverlapRatio(gt_rect, selected_dets)
[best_overlaps, good_ids] = max(ratio, [], 2)
iou = 0.4;
best_ids = good_ids(best_overlaps > iou)
pos_neg = zeros(size(selected_dets, 1), 1);
pos_neg(best_ids) = 1;
tp_boxes = selected_dets(pos_neg==1, :);
fp_boxes = selected_dets(pos_neg==0, :);
ann2 = insertObjectAnnotation(ann1, "rectangle", fp_boxes, "", "Color", 'red');
ann2 = insertObjectAnnotation(ann2, "rectangle", tp_boxes, "", "Color", 'green');
imshow(ann2) %[output:4982fad4]
%%
[P, R, sc] = calcpr(filtered_dets, filtered_scores, gt_rect, 0.4);
figure; %[output:95fdf52a]
plot(sc, P); %[output:95fdf52a]
hold on %[output:95fdf52a]
plot(sc, R); %[output:95fdf52a]
legend(["Precision", "Recall"]); %[output:95fdf52a]
xlabel("Score"); %[output:95fdf52a]
ylabel("%"); %[output:95fdf52a]
figure; %[output:5b3d9f54]
plot(R, P, 'x-'); %[output:5b3d9f54]
xlabel("Recall"); %[output:5b3d9f54]
ylabel("Precision"); %[output:5b3d9f54]
ap = trapz(R, P) %[output:2842ed82]
%%
function [P, R, sc] = calcpr(dets, scores, gt_rect, iou)
    cnt = size(scores, 2) + 1;
    P = zeros(cnt, 1);
    R = zeros(cnt, 1);
    sorted_scores = sort(scores);
    
    sorted_scores = [0, sorted_scores];
    sc = sorted_scores;
    
    all_pos = size(gt_rect, 1);
    
    P(1) = 1;
    R(1) = 0;
    
    for ii=2:cnt
        score_thr = sorted_scores(ii);
        selected_dets = dets(scores <= score_thr, :);
        ratio = bboxOverlapRatio(gt_rect, selected_dets);
        [best_overlaps, good_ids] = max(ratio, [], 2);
        best_ids = good_ids(best_overlaps > iou);
        tp = size(best_ids, 1);
        fn = all_pos - tp;
        fp = size(selected_dets, 1) - tp;
        P(ii) = tp / (tp + fp);
        R(ii) = tp / (tp + fn);
    end
end
