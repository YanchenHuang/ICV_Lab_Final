%% Image and Video Compression Lab (LMT@TUM)
% Lehrstuhl fuer Medientechnik
% Technische Universitaet Muenchen
% Kai CUI <kai.cui@tum.de> - SS2019
% Read the raw YUV sequences and convert them into RGB images
% Last modified 17.07.2019
%%
clc
clear
% Read a YUV file ,the file format is YUV420, which means the Cb and Cr are downsampled by a factor of 2
fid    = fopen('./BasketballDrive_1920_1080_50fps_420_8bit_1_20.yuv','rb'); % original sequence file
mkdir('./sequence_BasketballDrive_bmp') % target folder to save the images
width  = 1920; % image width
height = 1080; % image height
Frames = 20;   % number of frames to read
for i = 1 : Frames
    fprintf('The %d frame of %d frames\n', i, Frames)
    Y = fread(fid,[width, height],'uint8'); % read Y
    U = fread(fid,[width/2,height/2],'uint8'); % read CbCr
    V = fread(fid,[width/2,height/2],'uint8');
    %====in order to convert to RGB, CbCr are upsampled=========
    UU = uint8(imresize(U, 2)); % here we just use the built-in imresize function
    VV = uint8(imresize(V, 2));
    %=========================================================
    % transform the yuv444 to RGB, and save the images to bmp format
    YUV444 = cat(3,Y',UU',VV');
    RGB    = ycbcr2rgb(YUV444);
    Imagename = ['./sequence_BasketballDrive_bmp/BasketballDrive_',num2str(i-1, '%04d'),'.bmp'];
    imwrite(RGB, Imagename)
end