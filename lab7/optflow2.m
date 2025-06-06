% Read in a video file.
vidReader = VideoReader('visiontraffic.avi');

videoPlayer = vision.VideoPlayer();

% Create optical flow object.
opticFlow = opticalFlowFarneback;

% skip first still frames
for i=1:90
    frame = readFrame(vidReader);
end

% initialize optical flow
frameGray = rgb2gray(frame);

% estimate optical flow for the first frame (eliminates the noise for the
% actual first detection)
flow = estimateFlow(opticFlow,frameGray); 

ba = vision.BlobAnalysis;

figure(1);
tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact'); 

% Estimate the optical flow of objects in the video.
while hasFrame(vidReader)

    frameRGB = readFrame(vidReader);
    frameGray = rgb2gray(frameRGB);

    % estimate optical flow
    flow = estimateFlow(opticFlow,frameGray); 
    
    % show motion vectors
    nexttile(1)
    imshow(frameRGB * 0.3) 
    hold on
    plot(flow,'DecimationFactor',[15 15],'ScaleFactor',5)
    hold off 
    
    % calculate speed and direction from Vx and Vy
    dir = flow.Orientation;
    spd = flow.Magnitude;
    
    % show speed map
    nexttile(2)
    imshow(spd, [0, 10]);
    colormap(gca, 'jet');
    
    
    % threshold optical flow for speeds over 2
    nexttile(3)
    thr = spd > 2;\
    imshow(thr);
    
    % remove all measurements for speeds lower than 2
    filtdir = zeros(size(dir));
    filtdir(thr) = dir(thr);
    
    % calculate region statistics for thresholded image
    [AREA,CENTROID,BBOX] = step(ba, thr);
    
    % analyze found regions
    bb = [];
    lbl = [];
    cc = [];
    avgdir = [];
    avgspd = [];
    for i=1:size(AREA, 1)
        % leave only regions bigger than 2000 px
        if AREA(i) > 2000
            % extract the bounding box 
            x = BBOX(i,1);
            y = BBOX(i,2);
            w = BBOX(i,3);
            h = BBOX(i,4);
            
            % add the area to the list of labels
            lbl = [lbl AREA(i)];
            
            % store the bounding box
            bb = [bb; BBOX(i, :)];
            
            % compute the average direction inside the bounding box
            dirbb = filtdir(y:y+h-1,x:x+w-1);
            avgdir = [avgdir mean(dirbb(dirbb ~= 0))];
            
            spdbb = spd(y:y+h-1,x:x+w-1);
            avgspd = [avgspd mean(spdbb(spdbb > 2))];
            
            % store the centroid
            cc = [cc; CENTROID(i,:)];
        end
    end
    
    % draw annotations (bounding boxes with the segment area)
    ann = frameRGB;
    if size(bb, 1) > 0
        ann = insertObjectAnnotation(ann, 'rectangle', bb, lbl, 'TextBoxOpacity',0.9,'FontSize',18);
    end
    
    % show the annotated image
    nexttile(4)
    imshow(ann);
    hold on;
    
    % draw the average direction vector
    for i=1:size(bb,1)
        x1 = cc(i, 1);
        y1 = cc(i, 2);
        x2 = x1 + 5*avgspd(i)*cos(avgdir(i));
        y2 = y1 + 5*avgspd(i)*sin(avgdir(i));
        
        line([x1 x2], [y1 y2], 'LineWidth', 3);
    end
    hold off;
    pause(10^-2)
end