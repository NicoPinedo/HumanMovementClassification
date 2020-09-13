
function [output_flow] = process_flow(flow)
% Apply pre-defined image processing to input flow (modulus, gaussian filtering, open/close 
% morphing)

gaussian_sigma = 1;

flow = mod_flow(flow);
flow = imgaussfilt(flow, gaussian_sigma);
flow = im_morph(flow);

output_flow = flow;

end

function [output_flow] = mod_flow(flow)
% Return the modulus of input flow

flow(flow < 0) = -1*flow(flow < 0);
output_flow = flow;
end

function [output_image] = im_morph(image) 
% Apply open/close morphing to input image

se_open = strel("square", 3);
se_close = strel("square", 2);

image = imopen(image, se_open);
output_image = imclose(image, se_close);

end