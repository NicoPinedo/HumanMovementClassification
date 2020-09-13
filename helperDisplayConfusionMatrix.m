% Code taken from https://www.mathworks.com/help/vision/examples/digit-classification-using-hog-features.html
function helperDisplayConfusionMatrix(confMat)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

digits = '0':'5';
colHeadings = arrayfun(@(x)sprintf('%d',x),0:5,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'class  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
end