% define parameters
img = imread('/Users/Swa/Documents/NTU/Y3S1/CZ4003/OCR-Image-Binarization/samples/sample01.png'); 
imgname = rgb2gray(img);
filt_radius = 25; % filter radius [pixels]
k_threshold = 0.001; % std threshold parameter
% load the image
X = double(imgname); 
X = X / max(X(:)); % normalyze to [0, 1] range
% build filter
fgrid = -filt_radius : filt_radius;
[x, y] = meshgrid(fgrid);

filt = sqrt(x .^ 2 + y .^ 2) <= filt_radius;

filt = filt / sum(filt(:));

% calculate mean, and std
local_mean = imfilter(X, filt, 'symmetric');
local_std = sqrt(imfilter(X .^ 2, filt, 'symmetric'));
% calculate binary image
X_bin = X >= (local_mean + k_threshold * local_std);

imwrite(X_bin,'/Users/Swa/Documents/NTU/Y3S1/CZ4003/OCR-Image-Binarization/outputs/niblack.png');

% plot
figure; ax = zeros(4,1);
ax(1) = subplot(2,2,1); imshow(X); title('original image');
ax(2) = subplot(2,2,2); imshow(X_bin); title('binary image');
ax(3) = subplot(2,2,3); imshow(local_mean); title('local mean');
ax(4) = subplot(2,2,4); imshow(local_std); title('local std');
linkaxes(ax, 'xy');