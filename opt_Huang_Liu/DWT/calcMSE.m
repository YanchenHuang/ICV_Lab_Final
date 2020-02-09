    function MSE = calcMSE(Image, recImage)
    [height, width, cdim] = size(Image);
    MSE = sum(sum((double(Image) - double(recImage)).^2))/(height*width*cdim);
    MSE = sum(MSE(:));
    end