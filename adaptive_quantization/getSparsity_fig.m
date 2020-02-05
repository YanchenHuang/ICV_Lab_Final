%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure - Blockwise sparsity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all

% Parameters.
img=double(imread('lena.tif'));
img=rgb2gray(img/255);
block_nonzero=getSparsity(img);
PMF= hist(block_nonzero(:),min(block_nonzero(:)):max(block_nonzero(:)));
    
%%
figure;
subplot(1,2,1); % original image
imshow(img, []);
title('Original Image');

subplot(1,2,2);
imshow(block_nonzero', []);  
originalSize2 = get(gca, 'Position');
h = colorbar;
ylabel(h, 'k (number of non-zero elements)');
set(gca, 'Position', originalSize2);
title('Blockwise Sparsity');