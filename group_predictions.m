
function [newLabels] = group_predictions(predictionLabels, sample_count)

% Take mode of each sample feature vector for one video, and set all sample
% labels of that video to that mode.

label_count = size(predictionLabels, 1);
new_label_count = label_count / sample_count;
newLabels = string.empty(0, 1);

for r = 1:new_label_count 
    r_start = (r*sample_count) - sample_count + 1;
    r_end = r*sample_count;
    
    sample = predictionLabels(r_start:r_end, :);
    newLabels(r,:) = cell_mode(sample);
end

end