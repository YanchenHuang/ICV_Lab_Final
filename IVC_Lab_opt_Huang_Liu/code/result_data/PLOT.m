    load('adaptive_video.mat')
    load('dctvideo.mat')
    load('dwt_video.mat')
    
    figure(1)
    plot(dct_bitperpixel_av_video,dct_PSNR_av_video,'-*r');
    hold on;
    plot(bitperpixel_av_video_ad,PSNR_av_video_ad,'-*b');
     %hold on;

    legend('Baseline£¨DCT£©','adaptive qScale+deblock');
    title("DCT vs adaptive qScale+deblock)");
    axis.XLim = [0 4.5];
    axis.YLim = [24 42];
    axis.XTick = 0:0.1:4.5;
    axis.YTick = 24:2:42;
    xlabel('bpp');  
    ylabel('PSNR[dB]'); 
    
    figure(2)
    plot(dct_bitperpixel_av_video,dct_PSNR_av_video,'-*r');
    hold on;
    plot(dwt_bitperpixel_av_video,dwt_PSNR_av_video,'-*b');
    legend('Baseline£¨DCT£©','DWT');
    title("DCT vs DWT");
    axis.XLim = [0 4.5];
    axis.YLim = [24 42];
    axis.XTick = 0:0.1:4.5;
    axis.YTick = 24:2:42;
    xlabel('bpp');  
    ylabel('PSNR[dB]'); 