function [frame] = crop_frame(frame)

window_width = 50;
window_height = 100;

[y, x] = size(frame);
[Y, X] = ndgrid(1:y,1:x);
total_mass = sum(frame(:));

if total_mass == 0 
    frame = frame(1:101, 1:51);
    return 
end

x_centre = sum(X(:) .* frame(:)) / total_mass;
y_centre = sum(Y(:) .* frame(:)) / total_mass;

x_min = round(x_centre - (window_width/2));
x_max = round(x_centre + (window_width/2));
y_min = round(y_centre - (window_height/2));
y_max = round(y_centre + (window_height/2));

if x_min < 1
   overlap = (-1*x_min) + 1;
   
   overlap_mat = zeros(y, overlap);
   frame = [overlap_mat frame];
   
   x_min = 1;
   x_max = x_max + overlap; 
   
elseif x_max > x
   overlap = x_max - x;
   
   overlap_mat = zeros(y, overlap);
   frame = [frame overlap_mat]; 
   
end

% reinitialise x
x = size(frame, 2);

if y_min < 1
   overlap = (-1*y_min) + 1;
   
   overlap_mat = zeros(overlap, x);
   frame = [overlap_mat; frame];
   
   y_min = 1;
   y_max = y_max + overlap;  
   
elseif y_max > y
   overlap = y_max - y;
   
   overlap_mat = zeros(overlap, x);
   frame = [frame; overlap_mat];  
end

frame = frame(y_min:y_max, x_min:x_max);
if size(frame, 1) ~= 101 || size(frame, 2) ~= 51
    frame = frame(1:101, 1:51); % HOT FIX, due to rounding error ^
end
end