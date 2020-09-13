
% Load video data
load_videos();

% Start runtime timer
tic;

% Declare initial variables
opticFlow = opticalFlowLK();
class_strings = ["boxing" "handclapping" "handwaving" "jogging" "running" "walking"];
class_count = length(class_strings);
data_persons = 25;
data_scenarios = 4;
sample_frame_count = 48;
hog_cell = [10 10];
hogs_length = 5184; 
data_points_per_class = data_persons*data_scenarios*sample_frame_count;
data_points = data_persons*data_scenarios*sample_frame_count*class_count;
motion_threshold = [0.008 1];


% For each class
for class_str = class_strings
    
    disp("Extracting from " + class_str + " videos...")

    % Initialise hogs matrix for current class
    eval("hogs_" + class_str + " = single.empty(0, " + hogs_length + ");"); 
    
    hog_counter = 1;
    
    % Initialise video strings
    video_counter = 1;
    video_strings = string.empty(1, 0);
    for p = 1:data_persons
        for s = 1:data_scenarios
            video_strings(:, video_counter) = "data_" + class_str + "_p" + p + "_s" + s;
            video_counter = video_counter + 1;
        end
    end
    
    % For each video in class
    for video_str = video_strings

        % Get video by video_str
        video = eval(video_str);
        
        % If video exists
        if video ~= -1  
            max_frame = (video.FrameRate * floor(video.Duration)) - 3;
            % Get sample frame indices
            sample_frames = randi([1, max_frame], 1, sample_frame_count);

            % For each sample frame in video
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
                    flow_agg1 = flow_x1 + flow_y1;  % aggregate flow
                    
                    % Check if optical flows exceed threshold values
                    if (mean2(flow_agg1) < motion_threshold(1)) || (mean2(flow_agg1) > motion_threshold(2))
                        f = randi([1, max_frame], 1, 1);
                    else
                        motion_in_frame = true;
                    end
                end
                
                % Crop frames
                flow_x1 = crop_frame(flow_x1);
                flow_y1 = crop_frame(flow_y1);
                flow_x2 = crop_frame(flow_x2);
                flow_y2 = crop_frame(flow_y2);
               
                % Extract Histogram of Oriented Gradients (HOG) features
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

% Percentage of feature data to take as training/test data
train_percent = 0.9;
test_percent = 1 - train_percent;

% Initialise label arrays
trainingLabels = [];
testLabels = [];
trainingFeatures = [];
testFeatures = [];

% Partition feature data into training and test sets
disp("Partitioning and compling features, and creating class labels...")
for class_str = class_strings
    % eval function here, is used to execute strings as code
    eval("[index_max, dud] = size(hogs_" + class_str + ");");
    index_cut = train_percent*index_max;
    index_cut = round(index_cut / sample_frame_count) * sample_frame_count;
    index_cut2 = index_max - index_cut;
    eval("hogs_" + class_str + "_train = hogs_" + class_str + "(1:" + index_cut + ",:);");
    eval("hogs_" + class_str + "_test = hogs_" + class_str + "(" + (index_cut+1) + ":" + index_max + ",:);");
    
    eval("trainingFeatures = [trainingFeatures; hogs_" + class_str + "_train];");
    eval("testFeatures = [testFeatures; hogs_" + class_str + "_test];");
    
    trainingLabels = [trainingLabels; repmat(class_str, index_cut, 1)];
    testLabels = [testLabels; repmat(class_str, index_cut2, 1)];
    
    % Clear redundant variables 
    eval("clear hogs_" + class_str + "_train;");
    eval("clear hogs_" + class_str + "_test;");
    eval("clear hogs_" + class_str + ";");
end

% Perform Principal Component Analysis (PCA) to simplify training of classifier
% PCA code taken from https://www.mathworks.com/help/stats/pca.html
disp("PCA...")
[coeff,score,latent,tsquared,explained,mu] = pca(trainingFeatures);
sum_explained = 0;
idx = 0;
while sum_explained < 95
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end
scoreTrain95 = score(:,1:idx);
disp("Training classifier...")
classifier = fitcecoc(scoreTrain95, trainingLabels, 'FitPosterior', 1);
scoreTest95 = (testFeatures-mu)*coeff(:,1:idx);

% Predict test features
disp("Predicting test features...")
predictedLabels = predict(classifier, scoreTest95);
% Calculate classification loss
L = loss(classifier, scoreTest95, testLabels);

% Group predictions by video
g_predictedLabels = group_predictions(predictedLabels, sample_frame_count);
g_testLabels = group_predictions(testLabels, sample_frame_count);

% Tabulate results in a confusion matrix
disp("Tabulating results...")
conf_mat = confusionmat(g_testLabels, g_predictedLabels);
helperDisplayConfusionMatrix(conf_mat)

disp("Sample frames: " + sample_frame_count)
disp("HOG cell size: " + hog_cell)
disp("Loss: " + L)

% End runtime timer
toc;

 