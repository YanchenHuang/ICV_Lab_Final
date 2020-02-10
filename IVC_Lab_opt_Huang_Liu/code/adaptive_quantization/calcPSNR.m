function PSNR = calcPSNR(Image, recImage)
function MSE = calcMSE(Image, recImage)
[height, width, cdim] = size(Image);
MSE = sum(sum((double(Image) - double(recImage)).^2))/(height*width*cdim);
MSE = sum(MSE(:));
end
mse = calcMSE(Image, recImage);
PSNR = 20*log10(255/sqrt(mse));
end

