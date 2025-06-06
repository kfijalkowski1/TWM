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

% Initialize vehicle tracking variables
vehicleCount = 0;
trackedVehicles = [];  % Store centroids of tracked vehicles
vehicleTrajectories = {};  % Cell array to store trajectory for each vehicle (persistent)
allVehicleTrajectories = {};  % Store ALL trajectories including inactive ones
vehicleIDs = [];  % Store vehicle IDs
vehicleLastSeen = [];  % Frame counter for when vehicle was last seen
maxDistance = 100;     % Increased maximum distance to consider same vehicle between frames
maxMissedFrames = 20;  % Maximum frames a vehicle can be missing before considered gone
nextVehicleID = 1;     % Counter for assigning unique vehicle IDs
frameCount = 0;        % Frame counter

figure(1);
tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact'); 

% Estimate the optical flow of objects in the video.
while hasFrame(vidReader)
    frameCount = frameCount + 1;

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
    thr = (spd > 2) & (spd < 10);
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
    
    % Track unique vehicles based on centroids
    currentCentroids = cc;
    
    % Clean up vehicles that haven't been seen for too long
    % But preserve their trajectories in allVehicleTrajectories
    if ~isempty(trackedVehicles)
        activeVehicles = (frameCount - vehicleLastSeen) <= maxMissedFrames;
        
        % Store trajectories of vehicles being removed
        for i = 1:length(vehicleIDs)
            if ~activeVehicles(i)
                % Vehicle is being removed, save its trajectory permanently
                vehicleID = vehicleIDs(i);
                if vehicleID <= length(vehicleTrajectories) && ~isempty(vehicleTrajectories{vehicleID})
                    allVehicleTrajectories{vehicleID} = vehicleTrajectories{vehicleID};
                end
            end
        end
        
        % Remove inactive vehicles from active tracking
        trackedVehicles = trackedVehicles(activeVehicles, :);
        vehicleIDs = vehicleIDs(activeVehicles);
        vehicleLastSeen = vehicleLastSeen(activeVehicles);
    end
    
    if ~isempty(currentCentroids)
        if isempty(trackedVehicles)
            % First frame with vehicles - add all as new vehicles
            for i = 1:size(currentCentroids, 1)
                trackedVehicles = [trackedVehicles; currentCentroids(i, :)];
                vehicleIDs = [vehicleIDs; nextVehicleID];
                vehicleLastSeen = [vehicleLastSeen; frameCount];
                vehicleTrajectories{nextVehicleID} = currentCentroids(i, :);
                nextVehicleID = nextVehicleID + 1;
                vehicleCount = vehicleCount + 1;
            end
        else
            % Track existing vehicles and detect new ones
            matchedVehicles = false(size(currentCentroids, 1), 1);
            matchedTracked = false(size(trackedVehicles, 1), 1);
            
            % Use Hungarian algorithm approach - match each detection to closest vehicle
            for i = 1:size(currentCentroids, 1)
                currentCentroid = currentCentroids(i, :);
                
                % Calculate distances to all unmatched tracked vehicles
                availableVehicles = ~matchedTracked;
                if any(availableVehicles)
                    distances = sqrt(sum((trackedVehicles(availableVehicles, :) - currentCentroid).^2, 2));
                    availableIndices = find(availableVehicles);
                    
                    % Find closest available vehicle
                    [minDist, relativeIdx] = min(distances);
                    closestIdx = availableIndices(relativeIdx);
                    
                    if minDist <= maxDistance
                        % Update existing vehicle position and trajectory
                        trackedVehicles(closestIdx, :) = currentCentroid;
                        vehicleLastSeen(closestIdx) = frameCount;
                        vehicleID = vehicleIDs(closestIdx);
                        vehicleTrajectories{vehicleID} = [vehicleTrajectories{vehicleID}; currentCentroid];
                        matchedVehicles(i) = true;
                        matchedTracked(closestIdx) = true;
                    end
                end
            end
            
            % Add new vehicles that weren't matched
            for i = 1:size(currentCentroids, 1)
                if ~matchedVehicles(i)
                    trackedVehicles = [trackedVehicles; currentCentroids(i, :)];
                    vehicleIDs = [vehicleIDs; nextVehicleID];
                    vehicleLastSeen = [vehicleLastSeen; frameCount];
                    vehicleTrajectories{nextVehicleID} = currentCentroids(i, :);
                    nextVehicleID = nextVehicleID + 1;
                    vehicleCount = vehicleCount + 1;
                end
            end
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
    
    % Draw trajectory lines for all vehicles FIRST (so they're visible)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']; % Different colors for trajectories
    for vehicleID = 1:length(vehicleTrajectories)
        if ~isempty(vehicleTrajectories{vehicleID}) && size(vehicleTrajectories{vehicleID}, 1) > 1
            trajectory = vehicleTrajectories{vehicleID};
            colorIdx = mod(vehicleID - 1, length(colors)) + 1;
            plot(trajectory(:, 1), trajectory(:, 2), colors(colorIdx), 'LineWidth', 3, 'MarkerSize', 4);
            % Mark the start and end points
            plot(trajectory(1, 1), trajectory(1, 2), 'o', 'Color', colors(colorIdx), 'MarkerSize', 6, 'LineWidth', 2);
            plot(trajectory(end, 1), trajectory(end, 2), 's', 'Color', colors(colorIdx), 'MarkerSize', 6, 'LineWidth', 2);
        end
    end
    
    % draw the average direction vector on top of trajectories
    for i=1:size(bb,1)
        x1 = cc(i, 1);
        y1 = cc(i, 2);
        x2 = x1 + 5*avgspd(i)*cos(avgdir(i));
        y2 = y1 + 5*avgspd(i)*sin(avgdir(i));
        
        line([x1 x2], [y1 y2], 'LineWidth', 3, 'Color', 'k');
    end
    
    hold off;
    pause(10^-2)
end

% Print the total count of distinct vehicles detected
fprintf('Total distinct vehicles detected: %d\n', vehicleCount);