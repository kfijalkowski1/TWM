% Read in a video file.
vidReader = VideoReader('visiontraffic.avi');

% skip initial still frames
for i=1:90
    readFrame(vidReader);
end

detector = vision.ForegroundDetector(...
       'NumTrainingFrames', 5, ...
       'InitialVariance', 30*30);

ba = vision.BlobAnalysis;

figure(1);
tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact'); 

% Estimate the freground mask in the video.
while hasFrame(vidReader)
    frameRGB = readFrame(vidReader);
  
    fgMask = detector(frameRGB);
    fgMask = imclose(fgMask, strel('disk', 5));
      
    nexttile(2)
    imshow(fgMask);
        
    % calculate region statistics for the mask image
    [AREA,CENTROID,BBOX] = step(ba, fgMask);
    
    % analyze found regions
    bb = [];
    lbl = [];
    cc = [];
    for i=1:size(AREA, 1)
        % leave only regions bigger than 2000px
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
                        
            % store the centroid
            cc = [cc; CENTROID(i,:)];
        end
    end
    
    % draw annotations (bounding boxes with the segment area)
    ann = frameRGB;
    if size(bb, 1) > 0
        ann = insertObjectAnnotation(ann, 'rectangle', bb, lbl, 'TextBoxOpacity',0.9,'FontSize',18);
    end
    
    % show annotated image
    nexttile(1)
    imshow(ann);
    
    pause(0.01)
end