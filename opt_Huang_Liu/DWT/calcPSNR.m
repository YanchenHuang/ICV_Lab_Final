    function PSNR = calcPSNR(Image, recImage)
    mse = calcMSE(Image, recImage);
    PSNR = 20*log10(255/sqrt(mse));
    end