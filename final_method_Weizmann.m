
tic;

opticFlow = opticalFlowLK();
class_strings = ["running" "walking"];
class_count = length(class_strings);
data_persons = 10;
data_scenarios = 1;
sample_frame_count = 48;
hog_cell = [10 10];
hogs_length = 5184; 
data_points_per_class = data_persons*data_scenarios*sample_frame_count;
data_points = data_persons*data_scenarios*sample_frame_count*class_count;
motion_threshold = [0.008 1];

% for each class
for class_str = class_strings
    
    disp("Extracting from " + class_str + " videos...")

    % init hogs matrix for current class
    eval("hogs_" + class_str + " = single.empty(0, " + hogs_length + ");"); 
    
    hog_counter = 1;
    
    % init video strings
    video_counter = 1;
    video_strings = string.empty(1, 0);
    for p = (25+1):(25+data_persons)
        for s = 1:data_scenarios
            video_strings(:, video_counter) = "data_" + class_str + "_p" + p + "_s" + s;
            video_counter = video_counter + 1;
        end
    end
    
    % for each video in class
    for video_str = video_strings

        video = eval(video_str);
        
        if video ~= -1  % if video exists
            max_frame = (video.FrameRate * floor(video.Duration)) - 3;
            sample_frames = randi([1, max_frame], 1, sample_frame_count);

            % for each sample frame in video
            for f = sample_frames
                motion_in_frame = false;
                while motion_in_frame == false
                    frameRGB1 = read(video, f);
                    frameGray1 = rgb2gray(frameRGB1);
                    frameRGB2 = read(video, f+1);
                    frameGray2 = rgb2gray(frameRGB2);

                    flow1 = estimateFlow(opticFlow, frameGray1); 
                    flow_x1 = process_flow(flow1.Vx);
                    flow_y1 = process_flow(flow1.Vy);
                    flow2 = estimateFlow(opticFlow, frameGray2); 
                    flow_x2 = process_flow(flow2.Vx);
                    flow_y2 = process_flow(flow2.Vy);
                    flow_agg1 = flow_x1 + flow_y1;
                    
                    if (mean2(flow_agg1) < motion_threshold(1)) || (mean2(flow_agg1) > motion_threshold(2))
                        f = randi([1, max_frame], 1, 1);
                    else
                        motion_in_frame = true;
                    end
                end
                
                flow_x1 = crop_frame(flow_x1);
                flow_y1 = crop_frame(flow_y1);
                flow_x2 = crop_frame(flow_x2);
                flow_y2 = crop_frame(flow_y2);
               
                [hog_x1, vis_x1] = extractHOGFeatures(flow_x1, 'CellSize', hog_cell);
                [hog_y1, vis_y1] = extractHOGFeatures(flow_y1, 'CellSize', hog_cell);
                [hog_x2, vis_x2] = extractHOGFeatures(flow_x2, 'CellSize', hog_cell);
                [hog_y2, vis_y2] = extractHOGFeatures(flow_y2, 'CellSize', hog_cell);
                eval("hogs_" + class_str + "(hog_counter,:) = [hog_x1 hog_y1 hog_x2 hog_y2];");

                hog_counter = hog_counter + 1;
            end
        end
    end
end

weizLabels = [repmat("running", 10*sample_frame_count, 1); repmat("walking", 10*sample_frame_count, 1)];
weizFeatures = [hogs_running; hogs_walking];

% PCA
% PCA code adapted from https://www.mathworks.com/help/stats/pca.html
weizTest95 = (weizFeatures-mu)*coeff(:,1:idx);
weizPredictedLabels = predict(classifier, weizTest95);

% Group predictions by video
g_predictedLabels = group_predictions(weizPredictedLabels, sample_frame_count);
g_testLabels = group_predictions(weizLabels, sample_frame_count);

% Tabulate results
disp("Tabulating results...")
conf_mat = confusionmat(g_testLabels, g_predictedLabels);
helperDisplayConfusionMatrix(conf_mat)

disp("Sample frames: " + sample_frame_count)
disp("HOG cell size: " + hog_cell)

toc;

 