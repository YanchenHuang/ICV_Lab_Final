    figure
    plot(bitperpixel_av_still,PSNR_av_still,'-*r');
    hold on;
    plot(bitperpixel_av_still_ad,PSNR_av_still_ad,'-*b');
    hold on;
    plot(bitperpixel_av_still_ad_deblock,PSNR_av_still_ad_deblock,'-*g');
    hold on;
    plot(bitperpixel_av_still_ad_deblock,PSNR_av_still_ad_deblock,'-*k');
    hold on;
    legend('org','adpt',"org+deblock",'adapt+deblock');
    axis.XLim = [0 4.5];
    axis.YLim = [24 42];
    axis.XTick = 0:0.1:4.5;
    axis.YTick = 24:2:42;
    xlabel('bpp');  
    ylabel('PSNR[dB]'); 